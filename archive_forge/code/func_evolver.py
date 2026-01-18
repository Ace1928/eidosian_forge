from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import (
from pyrsistent._pmap import PMap, pmap
def evolver(self):
    """
        Returns an evolver of this object.
        """
    return _PRecordEvolver(self.__class__, self)