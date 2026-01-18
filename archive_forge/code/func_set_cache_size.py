import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
def set_cache_size(self, ignored):
    """Do nothing, we are passing through."""