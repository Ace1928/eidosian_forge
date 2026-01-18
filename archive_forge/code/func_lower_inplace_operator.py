import math
import sys
import itertools
from collections import namedtuple
import llvmlite.ir as ir
import numpy as np
import operator
from numba.np import arrayobj, ufunc_db, numpy_support
from numba.core.imputils import Registry, impl_ret_new_ref, force_error_model
from numba.core import typing, types, utils, cgutils, callconv
from numba.np.numpy_support import (
from numba.core.typing import npydecl
from numba.core.extending import overload, intrinsic
from numba.core import errors
from numba.cpython import builtins
def lower_inplace_operator(context, builder, sig, args):
    args = tuple(args) + (args[0],)
    sig = typing.signature(sig.return_type, *sig.args + (sig.args[0],))
    return numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel)