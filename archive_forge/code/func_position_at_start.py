import contextlib
import functools
from llvmlite.ir import instructions, types, values
def position_at_start(self, block):
    """
        Position at the start of the basic *block*.
        """
    self._block = block
    self._anchor = 0