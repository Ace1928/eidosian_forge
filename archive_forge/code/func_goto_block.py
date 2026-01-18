import contextlib
import functools
from llvmlite.ir import instructions, types, values
@contextlib.contextmanager
def goto_block(self, block):
    """
        A context manager which temporarily positions the builder at the end
        of basic block *bb* (but before any terminator).
        """
    old_block = self.basic_block
    term = block.terminator
    if term is not None:
        self.position_before(term)
    else:
        self.position_at_end(block)
    try:
        yield
    finally:
        self.position_at_end(old_block)