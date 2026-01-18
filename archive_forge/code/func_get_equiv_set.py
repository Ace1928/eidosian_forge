import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def get_equiv_set(self, block_label):
    """Return the equiv_set object of an block given its label.
        """
    return self.equiv_sets[block_label]