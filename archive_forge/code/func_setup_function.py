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
def setup_function(self, fndesc):
    self.function = self.context.declare_function(self.module, fndesc)
    if self.flags.dbg_optnone:
        attrset = self.function.attributes
        if 'alwaysinline' not in attrset:
            attrset.add('optnone')
            attrset.add('noinline')
    self.entry_block = self.function.append_basic_block('entry')
    self.builder = IRBuilder(self.entry_block)
    self.call_helper = self.call_conv.init_call_helper(self.builder)