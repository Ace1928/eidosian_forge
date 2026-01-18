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
def create_cuts_normal_vector(transBlock_rHull, var_info, hull_to_bigm_map, rBigM_linear_constraints, rHull_vars, disaggregated_vars, norm, cut_threshold, zero_tolerance, integer_arithmetic, constraint_tolerance):
    """Returns a cut which removes x* from the relaxed bigm feasible region.

    Ignores all parameters except var_info and cut_threshold, and constructs
    a cut at x_hat, the projection of the relaxed bigM solution x* onto the hull,
    which is perpendicular to the vector from x* to x_hat.

    Note that this method will often lead to numerical difficulties since both
    x* and x_hat are solutions to optimization problems. To mitigate this,
    use some method of backing off the cut to make it a bit more conservative.

    Parameters
    -----------
    transBlock_rHull: transformation block on relaxed hull instance. Ignored by
                      this callback.
    var_info: List of tuples (rBigM_var, rHull_var, xstar_param)
    hull_to_bigm_map: For expression substitution, maps id(hull_var) to
                      corresponding bigm var. Ignored by this callback
    rBigM_linear_constraints: list of linear constraints in relaxed bigM.
                              Ignored by this callback.
    rHull_vars: list of all variables in relaxed hull. Ignored by this callback.
    disaggregated_vars: ComponentSet of disaggregated variables in hull
                        reformulation. Ignored by this callback
    norm: The norm used in the separation problem, will be used to calculate
          the subgradient used to generate the cut
    cut_threshold: Amount x* needs to be infeasible in generated cut in order
                   to consider the cut for addition to the bigM model.
    zero_tolerance: Tolerance at which a float will be treated as 0 during
                    Fourier-Motzkin elimination. Ignored by this callback
    integer_arithmetic: Ignored by this callback (specifies FME use integer
                        arithmetic)
    constraint_tolerance: Ignored by this callback (specifies when constraints
                          are considered tight in FME)
    """
    cutexpr = 0
    if norm == 2:
        for x_rbigm, x_hull, x_star in var_info:
            cutexpr += (x_hull.value - x_star.value) * (x_rbigm - x_hull.value)
    elif norm == float('inf'):
        duals = transBlock_rHull.model().dual
        if len(duals) == 0:
            raise GDP_Error('No dual information in the separation problem! To use the infinity norm and the create_cuts_normal_vector method, you must use a solver which provides dual information.')
        i = 0
        for x_rbigm, x_hull, x_star in var_info:
            mu_plus = value(duals[transBlock_rHull.inf_norm_linearization[i]])
            mu_minus = value(duals[transBlock_rHull.inf_norm_linearization[i + 1]])
            assert mu_plus >= 0
            assert mu_minus >= 0
            cutexpr += (mu_plus - mu_minus) * (x_rbigm - x_hull.value)
            i += 2
    if value(cutexpr) < -cut_threshold:
        return [cutexpr >= 0]
    logger.warning('Generated cut did not remove relaxed BigM solution by more than the specified threshold of %s. Stopping cut generation.' % cut_threshold)
    return None