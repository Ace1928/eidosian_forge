from collections import Counter
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Constraint, Block
from pyomo.core.base.set import SetProduct
def get_indices_of_projection(index_set, *sets):
    """
    We want to project certain sets out of our indexing set.
    We return the parameterization of this projection (the product
    of remaining sets) and a function to recover the original index
    from an element of the parameterization and coordinates of the
    sets projected out.
    """
    s_set = ComponentSet(sets)
    try:
        total_s_dim = sum([s.dimen for s in sets])
    except TypeError:
        msg = 'get_indices_of_projection does not support sets with dimen == None, including those with inconsistent dimen'
        raise TypeError(msg)
    subset_set = ComponentSet(index_set.subsets())
    assert all((s in subset_set for s in sets))
    info = {}
    if isinstance(index_set, SetProduct):
        projection_sets = list(index_set.subsets())
        counter = Counter([id(_) for _ in projection_sets])
        for s in sets:
            if counter[id(s)] != 1:
                msg = 'Cannot omit sets that appear multiple times'
                raise ValueError(msg)
        location = {}
        other_ind_sets = []
        for ind_loc, ind_set in enumerate(projection_sets):
            found_set = False
            for s_loc, s_set in enumerate(sets):
                if ind_set is s_set:
                    location[ind_loc] = s_loc
                    found_set = True
                    break
            if not found_set:
                other_ind_sets.append(ind_set)
    else:
        location = {0: 0}
        other_ind_sets = []
    if index_set.dimen == total_s_dim:
        info['set_except'] = [None]
        info['index_getter'] = lambda incomplete_index, *newvals: newvals[0] if len(newvals) <= 1 else tuple([newvals[location[i]] for i in location])
        return info
    if len(other_ind_sets) == 1:
        set_except = other_ind_sets[0]
    elif len(other_ind_sets) >= 2:
        set_except = other_ind_sets[0].cross(*other_ind_sets[1:])
    else:
        raise ValueError('Did not expect this to happen')
    index_getter = lambda incomplete_index, *newvals: _complete_index(location, incomplete_index, *newvals)
    info['set_except'] = set_except
    info['index_getter'] = index_getter
    return info