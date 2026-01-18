import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
def enforce_number(n, flags):
    if flags.min_val is None and flags.max_val is None:
        return
    if not flags.min_val <= n <= flags.max_val:
        raise TypeError('bad number %s for type %s' % (str(n), flags.name))