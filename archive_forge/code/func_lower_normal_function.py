from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def lower_normal_function(self, fndesc):
    """
        Lower non-generator *fndesc*.
        """
    self.setup_function(fndesc)
    self.extract_function_arguments()
    entry_block_tail = self.lower_function_body()
    with debuginfo.suspend_emission(self.builder):
        self.builder.position_at_end(entry_block_tail)
        self.builder.branch(self.blkmap[self.firstblk])