import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def get_rel(self, name):
    """Retrieve a definition pair for the given variable,
        or return None if it is not available.
        """
    return guard(self._get_or_set_rel, name)