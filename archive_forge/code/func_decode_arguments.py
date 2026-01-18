from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def decode_arguments(self, builder, argtypes, func):
    """
        Get the decoded (unpacked) Python arguments with *argtypes*
        from LLVM function *func*.  A tuple of LLVM values is returned.
        """
    raw_args = self.get_arguments(func)
    arginfo = self._get_arg_packer(argtypes)
    return arginfo.from_arguments(builder, raw_args)