import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def insert_block(self, offset, scope=None, loc=None):
    scope = scope or self.current_scope
    loc = loc or self.loc
    blk = ir.Block(scope=scope, loc=loc)
    self.blocks[offset] = blk
    self.current_block = blk
    self.current_block_offset = offset
    return blk