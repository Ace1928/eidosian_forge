from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def get_cuid(self, key, context=None):
    """
        Get the time-indexed CUID corresponding to the provided key
        """
    return get_indexed_cuid(key, (self._orig_time_set,), context=context)