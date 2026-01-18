import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
@deprecated_method(deprecated_in((0, 7, 0)))
def deprecated_method(self):
    """Deprecated method docstring.

        This might explain stuff.
        """
    return 1