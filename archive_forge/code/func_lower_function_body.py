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
def lower_function_body(self):
    """
        Lower the current function's body, and return the entry block.
        """
    for offset in self.blocks:
        bname = 'B%s' % offset
        self.blkmap[offset] = self.function.append_basic_block(bname)
    self.pre_lower()
    entry_block_tail = self.builder.basic_block
    self.debug_print('# function begin: {0}'.format(self.fndesc.unique_name))
    for offset, block in sorted(self.blocks.items()):
        bb = self.blkmap[offset]
        self.builder.position_at_end(bb)
        self.debug_print(f'# lower block: {offset}')
        self.lower_block(block)
    self.post_lower()
    return entry_block_tail