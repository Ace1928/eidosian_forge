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
def lower_block(self, block):
    """
        Lower the given block.
        """
    self.pre_block(block)
    for inst in block.body:
        self.loc = inst.loc
        defaulterrcls = partial(LoweringError, loc=self.loc)
        with new_error_context('lowering "{inst}" at {loc}', inst=inst, loc=self.loc, errcls_=defaulterrcls):
            self.lower_inst(inst)
    self.post_block(block)