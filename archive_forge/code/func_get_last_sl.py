from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
def get_last_sl(self):
    """ Return last computed sparsity level by current scheduler.
        """
    return self._last_sl