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
def append_and_fix(x):
    """ Adds to the new_hole and fixes up definitions"""
    new_hole.append(x)
    if x.target.name in func_ir._definitions:
        assert len(func_ir._definitions[x.target.name]) == 1
        func_ir._definitions[x.target.name].clear()
    func_ir._definitions[x.target.name].append(x.value)