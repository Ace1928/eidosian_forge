import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
def alignof(self, key):
    """Get the specified alignment of the field.

        Since field alignment is optional, this may return None.
        """
    return self.fields[key].alignment