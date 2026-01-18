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
def fbbt_disjunct(disj, parent_bounds):
    orig_bnds = ComponentMap(parent_bounds)
    try:
        for var, var_bnds in disj._disj_var_bounds.items():
            scope_lb, scope_ub = var_bnds
            scope_lb = -inf if scope_lb is None else scope_lb
            scope_ub = inf if scope_ub is None else scope_ub
            parent_lb, parent_ub = parent_bounds.get(var, (-inf, inf))
            orig_bnds[var] = (max(scope_lb, parent_lb), min(scope_ub, parent_ub))
    except AttributeError:
        pass
    bnds_manager = BoundsManager(disj)
    bnds_manager.load_bounds(orig_bnds)
    try:
        new_bnds = fbbt(disj)
    except InfeasibleConstraintException as e:
        if disj.ctype == Disjunct:
            disj.deactivate()
        new_bnds = parent_bounds
    bnds_manager.pop_bounds()
    disj._disj_var_bounds = new_bnds
    for disj in disj.component_data_objects(Disjunct, active=True):
        fbbt_disjunct(disj, new_bnds)