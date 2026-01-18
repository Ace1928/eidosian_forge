import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
@intrinsic
def init_dummy(typingctx, data):

    def codegen(context, builder, sig, args):
        typ = sig.return_type
        data, = args
        dummy = cgutils.create_struct_proxy(typ)(context, builder)
        dummy.data = data
        if context.enable_nrt:
            context.nrt.incref(builder, sig.args[0], data)
        return dummy._getvalue()
    ret_typ = DummyType(data)
    sig = signature(ret_typ, data)
    return (sig, codegen)