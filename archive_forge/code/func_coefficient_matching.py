import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def coefficient_matching(model, constraint, uncertain_params, config):
    """
    :param model: master problem model
    :param constraint: the constraint from the master problem model
    :param uncertain_params: the list of uncertain parameters
    :param first_stage_variables: the list of effective first-stage variables (includes ssv if decision_rule_order = 0)
    :return: True if the coefficient matching was successful, False if its proven robust_infeasible due to
             constraints of the form 1 == 0
    """
    successful_matching = True
    robust_infeasible = False
    actual_uncertain_params = []
    for i in range(len(uncertain_params)):
        if not is_certain_parameter(uncertain_param_index=i, config=config):
            actual_uncertain_params.append(uncertain_params[i])
    if not hasattr(model, 'coefficient_matching_constraints'):
        model.coefficient_matching_constraints = ConstraintList()
    if not hasattr(model, 'swapped_constraints'):
        model.swapped_constraints = ConstraintList()
    variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
    params_in_constraint = ComponentSet(identify_mutable_parameters(constraint.expr))
    first_stage_variables = model.util.first_stage_variables
    second_stage_variables = model.util.second_stage_variables
    if all((v in ComponentSet(first_stage_variables) for v in variables_in_constraint)) and any((q in ComponentSet(actual_uncertain_params) for q in params_in_constraint)):
        pass
    elif all((v in ComponentSet(first_stage_variables + second_stage_variables) for v in variables_in_constraint)) and any((q in ComponentSet(actual_uncertain_params) for q in params_in_constraint)):
        constraint = substitute_ssv_in_dr_constraints(model=model, constraint=constraint)
        variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
        params_in_constraint = ComponentSet(identify_mutable_parameters(constraint.expr))
    else:
        pass
    if all((v in ComponentSet(first_stage_variables) for v in variables_in_constraint)) and any((q in ComponentSet(actual_uncertain_params) for q in params_in_constraint)):
        model.param_set = []
        for i in range(len(list(variables_in_constraint))):
            model.add_component('p_%s' % i, Param(initialize=1, mutable=True))
            model.param_set.append(getattr(model, 'p_%s' % i))
        model.variable_set = []
        for i in range(len(list(actual_uncertain_params))):
            model.add_component('x_%s' % i, Var(initialize=1))
            model.variable_set.append(getattr(model, 'x_%s' % i))
        original_var_to_param_map = list(zip(list(variables_in_constraint), model.param_set))
        original_param_to_vap_map = list(zip(list(actual_uncertain_params), model.variable_set))
        var_to_param_substitution_map_forward = {}
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_forward[id(var)] = param
        param_to_var_substitution_map_forward = {}
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_forward[id(param)] = var
        var_to_param_substitution_map_reverse = {}
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_reverse[id(param)] = var
        param_to_var_substitution_map_reverse = {}
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_reverse[id(var)] = param
        model.swapped_constraints.add(replace_expressions(expr=replace_expressions(expr=constraint.lower, substitution_map=param_to_var_substitution_map_forward), substitution_map=var_to_param_substitution_map_forward) == replace_expressions(expr=replace_expressions(expr=constraint.body, substitution_map=param_to_var_substitution_map_forward), substitution_map=var_to_param_substitution_map_forward))
        swapped = model.swapped_constraints[max(model.swapped_constraints.keys())]
        val = generate_standard_repn(swapped.body, compute_values=False)
        if val.constant is not None:
            if type(val.constant) not in native_types:
                temp_expr = replace_expressions(val.constant, substitution_map=var_to_param_substitution_map_reverse)
                temp_expr = generate_standard_repn(temp_expr).to_expression()
                if temp_expr.__class__ not in native_types:
                    model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
            elif math.isclose(value(val.constant), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                pass
            else:
                successful_matching = False
                robust_infeasible = True
        if val.linear_coefs is not None:
            for coeff in val.linear_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(coeff, substitution_map=var_to_param_substitution_map_reverse)
                    temp_expr = generate_standard_repn(temp_expr).to_expression()
                    if temp_expr.__class__ not in native_types:
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(value(coeff), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.quadratic_coefs:
            for coeff in val.quadratic_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(coeff, substitution_map=var_to_param_substitution_map_reverse)
                    temp_expr = generate_standard_repn(temp_expr).to_expression()
                    if temp_expr.__class__ not in native_types:
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(value(coeff), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.nonlinear_expr is not None:
            successful_matching = False
            robust_infeasible = False
        if successful_matching:
            model.util.h_x_q_constraints.add(constraint)
    for i in range(len(list(variables_in_constraint))):
        model.del_component('p_%s' % i)
    for i in range(len(list(params_in_constraint))):
        model.del_component('x_%s' % i)
    model.del_component('swapped_constraints')
    model.del_component('swapped_constraints_index')
    return (successful_matching, robust_infeasible)