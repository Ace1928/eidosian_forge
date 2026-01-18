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
def get_assignment_source(self, destname):
    """
        Get a possible assignment source (a ir.Var instance) to replace
        *destname*, otherwise None.
        """
    if destname in self.dest_to_src:
        return self.dest_to_src[destname]
    self.unused_dests.discard(destname)
    return None