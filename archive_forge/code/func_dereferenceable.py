import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@dereferenceable.setter
def dereferenceable(self, val):
    assert isinstance(val, int) and val >= 0
    self._dereferenceable = val