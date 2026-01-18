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
def raise_missing_name_error(self, name):
    msg = "name '%s' is not defined" % name
    cstr = self.context.insert_const_string(self.module, msg)
    self.err_set_string('PyExc_NameError', cstr)