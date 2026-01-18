import contextlib
import functools
from llvmlite.ir import instructions, types, values
def cbranch(self, cond, truebr, falsebr):
    """
        Conditional branch to *truebr* if *cond* is true, else to *falsebr*.
        """
    br = instructions.ConditionalBranch(self.block, 'br', [cond, truebr, falsebr])
    self._set_terminator(br)
    return br