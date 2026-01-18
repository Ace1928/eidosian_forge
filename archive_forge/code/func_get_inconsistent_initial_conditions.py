from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block, value
from pyomo.dae.set_utils import (
def get_inconsistent_initial_conditions(model, time, tol=1e-08, t0=None, allow_skip=True, suppress_warnings=False):
    """Finds constraints of the model that are implicitly or explicitly
    indexed by time and checks if they are consistent to within a tolerance
    at the initial value of time.

    Args:
        model: Model whose constraints to check
        time: Set whose initial condition will be checked
        tol: Maximum constraint violation
        t0: Point in time at which to check constraints

    Returns:
        List of constraint data objects that were found to be inconsistent.
    """
    if t0 is None:
        t0 = time.first()
    inconsistent = ComponentSet()
    for con in model.component_objects(Constraint, active=True):
        if not is_explicitly_indexed_by(con, time):
            continue
        if is_in_block_indexed_by(con, time):
            continue
        info = get_index_set_except(con, time)
        non_time_set = info['set_except']
        index_getter = info['index_getter']
        for non_time_index in non_time_set:
            index = index_getter(non_time_index, t0)
            try:
                condata = con[index]
            except KeyError:
                if not suppress_warnings:
                    print(index_warning(con.name, index))
                if not allow_skip:
                    raise
                continue
            if value(condata.body) - value(condata.upper) > tol or value(condata.lower) - value(condata.body) > tol:
                inconsistent.add(condata)
    for blk in model.component_objects(Block, active=True):
        if not is_explicitly_indexed_by(blk, time):
            continue
        if is_in_block_indexed_by(blk, time):
            continue
        info = get_index_set_except(blk, time)
        non_time_set = info['set_except']
        index_getter = info['index_getter']
        for non_time_index in non_time_set:
            index = index_getter(non_time_index, t0)
            blkdata = blk[index]
            for condata in blkdata.component_data_objects(Constraint, active=True):
                if value(condata.body) - value(condata.upper) > tol or value(condata.lower) - value(condata.body) > tol:
                    if condata in inconsistent:
                        raise ValueError('%s has already been visited. The only way this should happen is if the model has nested time-indexed blocks, which is not supported.')
                    inconsistent.add(condata)
    return list(inconsistent)