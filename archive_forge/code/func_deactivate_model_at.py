from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def deactivate_model_at(b, cset, pts, allow_skip=True, suppress_warnings=False):
    """
    Finds any block or constraint in block b, indexed explicitly (and not
    implicitly) by cset, and deactivates it at points specified.
    Implicitly indexed components are excluded because one of their parent
    blocks will be deactivated, so deactivating them too would be redundant.

    Args:
        b : Block to search
        cset : ContinuousSet of interest
        pts : Value or list of values, in ContinuousSet, to deactivate at

    Returns:
        A dictionary mapping points in pts to lists of
        component data that have been deactivated there
    """
    if type(pts) is not list:
        pts = [pts]
    for pt in pts:
        if pt not in cset:
            msg = str(pt) + ' is not in ContinuousSet ' + cset.name
            raise ValueError(msg)
    deactivated = {pt: [] for pt in pts}
    visited = set()
    for comp in b.component_objects([Block, Constraint], active=True):
        if id(comp) in visited:
            continue
        visited.add(id(comp))
        if is_explicitly_indexed_by(comp, cset) and (not is_in_block_indexed_by(comp, cset)):
            info = get_index_set_except(comp, cset)
            non_cset_set = info['set_except']
            index_getter = info['index_getter']
            for non_cset_index in non_cset_set:
                for pt in pts:
                    index = index_getter(non_cset_index, pt)
                    try:
                        comp[index].deactivate()
                        deactivated[pt].append(comp[index])
                    except KeyError:
                        if not suppress_warnings:
                            print(index_warning(comp.name, index))
                        if not allow_skip:
                            raise
                        continue
    return deactivated