import contextlib
import functools
from llvmlite.ir import instructions, types, values
def insert_element(self, vector, value, idx, name=''):
    """
        Returns vector with vector[idx] replaced by value.
        The result is undefined if the idx is larger or equal the vector length.
        """
    instr = instructions.InsertElement(self.block, vector, value, idx, name=name)
    self._insert(instr)
    return instr