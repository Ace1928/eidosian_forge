from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def clear_all(self):
    """
        Clears the entire cache. To be used when some event results in unknown
        invalidations across this particular `DataLoader`. Returns itself for
        method chaining.
        """
    self._promise_cache.clear()
    return self