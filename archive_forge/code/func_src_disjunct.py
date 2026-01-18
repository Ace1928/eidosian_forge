from pyomo.common.autoslots import AutoSlots
from pyomo.core.base.block import _BlockData, IndexedBlock
from pyomo.core.base.global_set import UnindexedComponent_index, UnindexedComponent_set
@property
def src_disjunct(self):
    return None if self._src_disjunct is None else self._src_disjunct()