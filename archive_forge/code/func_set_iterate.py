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
@contextlib.contextmanager
def set_iterate(self, set):
    builder = self.builder
    hashptr = cgutils.alloca_once(builder, self.py_hash_t, name='hashptr')
    keyptr = cgutils.alloca_once(builder, self.pyobj, name='keyptr')
    posptr = cgutils.alloca_once_value(builder, Constant(self.py_ssize_t, 0), name='posptr')
    bb_body = builder.append_basic_block('bb_body')
    bb_end = builder.append_basic_block('bb_end')
    builder.branch(bb_body)

    def do_break():
        builder.branch(bb_end)
    with builder.goto_block(bb_body):
        r = self.set_next_entry(set, posptr, keyptr, hashptr)
        finished = cgutils.is_null(builder, r)
        with builder.if_then(finished, likely=False):
            builder.branch(bb_end)
        yield _IteratorLoop(builder.load(keyptr), do_break)
        builder.branch(bb_body)
    builder.position_at_end(bb_end)