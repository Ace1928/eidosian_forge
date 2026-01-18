import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
def register_clean(self, an_object, precious=False):
    """Register an_object as being clean.

        Note that precious is only a hint, and PassThroughTransaction
        ignores it.
        """