from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def log_close_to_bounds(m, tol=1e-06, logger=logger):
    """Print the variables and constraints that are near their bounds.

    See :py:func:`find_close_to_bounds()` for a description of the
    variables and constraints that are returned (and which are omitted).

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    logger: logging.Logger
        Logger to output to; defaults to `pyomo.util.infeasible`.

    """
    if logger.getEffectiveLevel() > logging.INFO:
        logger.warning('log_close_to_bounds() called with a logger whose effective level is higher than logging.INFO: no output will be logged regardless of bound status')
    for obj, val, close in find_close_to_bounds(m, tol):
        if not close:
            if obj.ctype is Var:
                logger.debug(f'Skipping VAR {obj.name} with no assigned value.')
            elif obj.ctype is Constraint:
                logger.info(f'Skipping CONSTR {obj.name}: evaluation error.')
            else:
                logger.error(f'Object {obj.name} was neither a Var nor Constraint')
            continue
        if close & 1:
            logger.info(f'{obj.name} near LB of {obj.lb}')
        if close & 2:
            logger.info(f'{obj.name} near UB of {obj.ub}')
    return