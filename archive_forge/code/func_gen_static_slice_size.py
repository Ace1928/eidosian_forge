import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def gen_static_slice_size(self, lhs_rel, rhs_rel, loc, scope, stmts, equiv_set):
    the_var, *_ = self.gen_literal_slice_part(rhs_rel - lhs_rel, loc, scope, stmts, equiv_set, name='static_slice_size')
    return the_var