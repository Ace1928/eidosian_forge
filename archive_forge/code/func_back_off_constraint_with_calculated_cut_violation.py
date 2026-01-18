from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def back_off_constraint_with_calculated_cut_violation(cut, transBlock_rHull, bigm_to_hull_map, opt, stream_solver, TOL):
    """Calculates the maximum violation of cut subject to the relaxed hull
    constraints. Increases this violation by TOL (to account for optimality
    tolerance in solving the problem), and, if it finds that cut can be violated
    up to this tolerance, makes it more conservative such that it no longer can.

    Parameters
    ----------
    cut: The cut to be made more conservative, a Constraint
    transBlock_rHull: the relaxed hull model's transformation Block
    bigm_to_hull_map: Dictionary mapping ids of bigM variables to the
                      corresponding variables on the relaxed hull instance
    opt: SolverFactory object for solving the maximum violation problem
    stream_solver: Whether or not to set tee=True while solving the maximum
                   violation problem.
    TOL: An absolute tolerance to be added to the calculated cut violation,
         to account for optimality tolerance in the maximum violation problem
         solve.
    """
    instance_rHull = transBlock_rHull.model()
    logger.info('Post-processing cut: %s' % cut.expr)
    transBlock_rHull.separation_objective.deactivate()
    transBlock_rHull.infeasibility_objective = Objective(expr=clone_without_expression_components(cut.body, substitute=bigm_to_hull_map))
    results = opt.solve(instance_rHull, tee=stream_solver, load_solutions=False)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning('Problem to determine how much to back off the new cut did not solve normally. Leaving the constraint as is, which could lead to numerical trouble%s' % (results,))
        transBlock_rHull.del_component(transBlock_rHull.infeasibility_objective)
        transBlock_rHull.separation_objective.activate()
        return
    instance_rHull.solutions.load_from(results)
    val = value(transBlock_rHull.infeasibility_objective) - TOL
    if val <= 0:
        logger.info('\tBacking off cut by %s' % val)
        cut._body += abs(val)
    transBlock_rHull.del_component(transBlock_rHull.infeasibility_objective)
    transBlock_rHull.separation_objective.activate()