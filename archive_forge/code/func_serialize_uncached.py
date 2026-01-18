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
def serialize_uncached(self, obj):
    """
        Same as serialize_object(), but don't create a global variable,
        simply return a literal for structure:
        {i8* data, i32 length, i8* hashbuf, i8* func_ptr, i32 alloc_flag}
        """
    data = serialize.dumps(obj)
    assert len(data) < 2 ** 31
    name = '.const.pickledata.%s' % (id(obj) if config.DIFF_IR == 0 else 'DIFF_IR')
    bdata = cgutils.make_bytearray(data)
    hashed = cgutils.make_bytearray(hashlib.sha1(data).digest())
    arr = self.context.insert_unique_const(self.module, name, bdata)
    hasharr = self.context.insert_unique_const(self.module, f'{name}.sha1', hashed)
    struct = Constant.literal_struct([arr.bitcast(self.voidptr), Constant(ir.IntType(32), arr.type.pointee.count), hasharr.bitcast(self.voidptr), cgutils.get_null_value(self.voidptr), Constant(ir.IntType(32), 0)])
    return struct