import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def create_resumption_block(self, lower, index):
    block_name = 'generator_resume%d' % (index,)
    block = lower.function.append_basic_block(block_name)
    lower.builder.position_at_end(block)
    self.resume_blocks[index] = block