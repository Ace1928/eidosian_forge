from pyomo.core.base import (
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR, enforce_dr_degree
def construct_master_feasibility_problem(model_data, config):
    """
    Construct a slack-variable based master feasibility model.
    Initialize all model variables appropriately, and scale slack variables
    as well.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    model : ConcreteModel
        Slack variable model.
    """
    model = model_data.master_model.clone()
    varmap_name = unique_component_name(model_data.master_model, 'pyros_var_map')
    setattr(model_data.master_model, varmap_name, list(model_data.master_model.component_data_objects(Var)))
    model = model_data.master_model.clone()
    model_data.feasibility_problem_varmap = list(zip(getattr(model_data.master_model, varmap_name), getattr(model, varmap_name)))
    delattr(model_data.master_model, varmap_name)
    delattr(model, varmap_name)
    for obj in model.component_data_objects(Objective):
        obj.deactivate()
    iteration = model_data.iteration
    targets = []
    for blk in model.scenarios[iteration, :]:
        targets.extend([con for con in blk.component_data_objects(Constraint, active=True, descend_into=True) if not con.equality])
    pre_slack_con_exprs = ComponentMap(((con, con.body - con.upper) for con in targets))
    TransformationFactory('core.add_slack_variables').apply_to(model, targets=targets)
    slack_vars = ComponentSet(model._core_add_slack_variables.component_data_objects(Var, descend_into=True))
    for con in pre_slack_con_exprs:
        repn = generate_standard_repn(con.body)
        slack_var_coef_map = ComponentMap()
        for idx in range(len(repn.linear_vars)):
            var = repn.linear_vars[idx]
            if var in slack_vars:
                slack_var_coef_map[var] = repn.linear_coefs[idx]
        slack_substitution_map = dict()
        for slack_var in slack_var_coef_map:
            if slack_var_coef_map[slack_var] == -1:
                con_slack = max(0, value(pre_slack_con_exprs[con]))
            else:
                con_slack = max(0, -value(pre_slack_con_exprs[con]))
            slack_var.set_value(con_slack)
            scaling_coeff = 1
            slack_substitution_map[id(slack_var)] = scaling_coeff * slack_var
        con.set_value((replace_expressions(con.lower, slack_substitution_map), replace_expressions(con.body, slack_substitution_map), replace_expressions(con.upper, slack_substitution_map)))
    return model