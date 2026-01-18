from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from collections import OrderedDict
def generate_sliced_components(b, index_stack, slice_, sets, ctype, index_map, active=None):
    """Recursively generate slices of the specified ctype along the
    specified sets

    Parameters
    ----------

    b: _BlockData
        Block whose components will be sliced

    index_stack: list
        Sets above ``b`` in the block hierarchy, including on its parent
        component, that have been sliced. This is necessary to return the
        sets that have been sliced.

    slice_: IndexedComponent_slice or _BlockData
        Slice generated so far.  This function will yield extensions to
        this slice at the current level of the block hierarchy.

    sets: ComponentSet of Pyomo sets
        Sets that will be sliced

    ctype: Subclass of Component
        Type of components to generate

    index_map: ComponentMap
        Map from (some of) the specified sets to a "representative index"
        to use when descending into subblocks. While this map does not need
        to contain every set in the sliced sets, it must not contain any
        sets that will not be sliced.

    active: Bool or None
        If not None, this is a boolean flag used to filter component objects
        by their active status.

    Yields
    ------

    Tuple of Sets and an IndexedComponent_slice or ComponentData
        The sets indexing the returned component or slice. If the component
        is indexed, an IndexedComponent_slice is returned. Otherwise, a
        ComponentData is returned.

    """
    if type(slice_) is IndexedComponent_slice:
        context_slice = slice_.duplicate()
    else:
        context_slice = None
    if active is not None and active != b.active:
        return
    check_active = issubclass(ctype, ActiveComponent) and active != None
    c_active = active if check_active else None
    for c in b.component_objects(ctype, descend_into=False, active=c_active):
        subsets = list(c.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]
        sliced_sets = index_stack + new_sets
        for idx, new_slice in slice_component_along_sets(c, sets, context_slice=context_slice, normalize=False):
            if not check_active or (not sliced_sets and new_slice.active == c_active) or (sliced_sets and any((data.active == c_active for data in new_slice.duplicate()))):
                yield (sliced_sets, new_slice)
    for sub in b.component_objects(Block, descend_into=False, active=active):
        subsets = list(sub.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]
        index_stack.extend(new_sets)
        given_descend_idx = [_NotAnIndex for _ in subsets]
        for i, s in enumerate(subsets):
            if s in index_map:
                given_descend_idx[i] = index_map[s]
                if s not in sets:
                    raise RuntimeError('Encountered a specified index for a set %s that we are not slicing. This is not supported' % s)
            elif s in sets:
                given_descend_idx[i] = get_slice_for_set(s)
        for idx, new_slice in slice_component_along_sets(sub, sets, context_slice=context_slice, normalize=False):
            if sub.is_indexed():
                descend_idx = _fill_indices(list(given_descend_idx), idx)
                descend_data = sub[descend_idx]
                if type(descend_data) is IndexedComponent_slice:
                    try:
                        slice_iter = iter(descend_data)
                        _data = next(slice_iter)
                        while active is not None and _data.active != active:
                            _data = next(slice_iter)
                        descend_data = _data
                    except StopIteration:
                        continue
                elif active is not None and descend_data.active != active:
                    continue
            else:
                descend_data = sub
            for st, v in generate_sliced_components(descend_data, index_stack, new_slice, sets, ctype, index_map, active=active):
                yield (tuple(st), v)
        for _ in new_sets:
            index_stack.pop()