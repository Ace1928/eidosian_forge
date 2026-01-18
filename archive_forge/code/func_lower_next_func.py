import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def lower_next_func(self, lower):
    """
        Lower the generator's next() function (which takes the
        passed-by-reference generator structure and returns the next
        yielded value).
        """
    lower.setup_function(self.gendesc)
    lower.debug_print('# lower_next_func: {0}'.format(self.gendesc.unique_name))
    assert self.gendesc.argtypes[0] == self.gentype
    builder = lower.builder
    function = lower.function
    genptr, = self.call_conv.get_arguments(function)
    self.arg_packer.load_into(builder, self.get_args_ptr(builder, genptr), lower.fnargs)
    self.resume_index_ptr = self.get_resume_index_ptr(builder, genptr)
    self.gen_state_ptr = self.get_state_ptr(builder, genptr)
    prologue = function.append_basic_block('generator_prologue')
    entry_block_tail = lower.lower_function_body()
    stop_block = function.append_basic_block('stop_iteration')
    builder.position_at_end(stop_block)
    self.call_conv.return_stop_iteration(builder)
    builder.position_at_end(prologue)
    first_block = self.resume_blocks[0] = lower.blkmap[lower.firstblk]
    switch = builder.switch(builder.load(self.resume_index_ptr), stop_block)
    for index, block in self.resume_blocks.items():
        switch.add_case(index, block)
    builder.position_at_end(entry_block_tail)
    builder.branch(prologue)