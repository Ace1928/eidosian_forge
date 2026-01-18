from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from collections import OrderedDict
def get_slice_for_set(s):
    """
    Get the slice of the proper dimension for a set.
    """
    if s.dimen != 0:
        if not normalize_index.flatten:
            return slice(None)
        elif s.dimen is not None:
            return (slice(None),) * s.dimen
        else:
            return (Ellipsis,)
    else:
        return None