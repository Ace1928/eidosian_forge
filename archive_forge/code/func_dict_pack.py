from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def dict_pack(self, keyvalues):
    """
        Args
        -----
        keyvalues: iterable of (str, llvm.Value of PyObject*)
        """
    dictobj = self.dict_new()
    with self.if_object_ok(dictobj):
        for k, v in keyvalues:
            self.dict_setitem_string(dictobj, k, v)
    return dictobj