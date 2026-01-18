import contextlib
import functools
from llvmlite.ir import instructions, types, values
def position_before(self, instr):
    """
        Position immediately before the given instruction.  The current block
        is also changed to the instruction's basic block.
        """
    self._block = instr.parent
    self._anchor = self._block.instructions.index(instr)