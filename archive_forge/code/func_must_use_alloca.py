import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def must_use_alloca(blocks):
    """
    Analyzes a dictionary of blocks to find variables that must be
    stack allocated with alloca.  For each statement in the blocks,
    determine if that statement requires certain variables to be
    stack allocated.  This function uses the extension point
    ir_extension_use_alloca to allow other IR node types like parfors
    to register to be processed by this analysis function.  At the
    moment, parfors are the only IR node types that may require
    something to be stack allocated.
    """
    use_alloca_vars = set()
    for ir_block in blocks.values():
        for stmt in ir_block.body:
            if type(stmt) in ir_extension_use_alloca:
                func = ir_extension_use_alloca[type(stmt)]
                func(stmt, use_alloca_vars)
                continue
    return use_alloca_vars