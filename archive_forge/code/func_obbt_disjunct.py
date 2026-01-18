from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt, BoundsManager
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr import identify_variables
from pyomo.core import Constraint, Objective, TransformationFactory, minimize, value
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
def obbt_disjunct(orig_model, idx, solver):
    model = orig_model.clone()
    disjunct = model._disjuncts_to_process[idx]
    disjunct.indicator_var.fix(1)
    for obj in model.component_data_objects(Objective, active=True):
        obj.deactivate()
    for constr in model.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() not in linear_degrees:
            constr.deactivate()
    relevant_var_set = ComponentSet()
    for constr in disjunct.component_data_objects(Constraint, active=True):
        relevant_var_set.update(identify_variables(constr.body, include_fixed=False))
    TransformationFactory('gdp.bigm').apply_to(model)
    model._var_bounding_obj = Objective(expr=1, sense=minimize)
    for var in relevant_var_set:
        model._var_bounding_obj.set_value(expr=var)
        var_lb = solve_bounding_problem(model, solver)
        if var_lb is None:
            return None
        model._var_bounding_obj.set_value(expr=-var)
        var_ub = solve_bounding_problem(model, solver)
        if var_ub is None:
            return None
        else:
            var_ub = -var_ub
        var.setlb(var_lb)
        var.setub(var_ub)
    var_bnds = ComponentMap(((orig_var, (clone_var.lb if clone_var.has_lb() else -inf, clone_var.ub if clone_var.has_ub() else inf)) for orig_var, clone_var in zip(orig_model._disj_bnds_linear_vars, model._disj_bnds_linear_vars) if clone_var in relevant_var_set))
    return var_bnds