import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def remove_redefineds(self, redefineds):
    """Take a set of variables in redefineds and go through all
        the currently existing equivalence sets (created in topo order)
        and remove that variable from all of them since it is multiply
        defined within the function.
        """
    unused = set()
    for r in redefineds:
        for eslabel in self.equiv_sets:
            es = self.equiv_sets[eslabel]
            es.define(r, unused)