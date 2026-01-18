from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block, value
from pyomo.dae.set_utils import (
def solve_consistent_initial_conditions(model, time, solver, tee=False, allow_skip=True, suppress_warnings=False):
    """
    Solves a model with all Constraints and Blocks deactivated except
    at the initial value of the Set time. Reactivates Constraints and
    Blocks that got deactivated.

    Args:
        model: Model that will be solved
        time: Set whose initial conditions will remain active for solve
        solver: Something that implements a solve method that accepts
                a model and tee keyword as arguments
        tee: tee argument that will be sent to solver's solve method
        allow_skip: If True, KeyErrors due to Constraint.Skip being
                    used will be ignored
        suppress_warnings: If True, warnings due to ignored
                           KeyErrors will be suppressed

    Returns:
        The object returned by the solver's solve method
    """
    scheme = time.get_discretization_info()['scheme']
    if scheme != 'LAGRANGE-RADAU' and scheme != 'BACKWARD Difference':
        raise NotImplementedError('%s discretization scheme is not supported' % scheme)
    timelist = list(time)[1:]
    deactivated_dict = deactivate_model_at(model, time, timelist, allow_skip=allow_skip, suppress_warnings=suppress_warnings)
    result = solver.solve(model, tee=tee)
    for t in timelist:
        for comp in deactivated_dict[t]:
            comp.activate()
    return result