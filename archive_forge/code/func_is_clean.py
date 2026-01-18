import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
def is_clean(self, an_object):
    """Return True if an_object is clean."""
    return an_object in self._clean_objects