import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
def descr_body(self, buf):
    """
        Describe of the body of the function.
        """
    for blk in self.blocks:
        blk.descr(buf)