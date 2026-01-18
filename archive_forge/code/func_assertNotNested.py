from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def assertNotNested(self):
    """
        Check that no other objects are being built while making this
        object. If not, raise an exception.
        """
    if self.nested:
        raise IsNestedError()