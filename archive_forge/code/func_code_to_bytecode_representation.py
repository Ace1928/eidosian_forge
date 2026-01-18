import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def code_to_bytecode_representation(co, use_func_first_line=False):
    """
    A simple disassemble of bytecode.

    It does not attempt to provide the full Python source code, rather, it provides a low-level
    representation of the bytecode, respecting the lines (so, its target is making the bytecode
    easier to grasp and not providing the original source code).

    Note that it does show jump locations/targets and converts some common bytecode constructs to
    Python code to make it a bit easier to understand.
    """
    if use_func_first_line:
        firstlineno = co.co_firstlineno
    else:
        firstlineno = 0
    return _Disassembler(co, firstlineno).disassemble()