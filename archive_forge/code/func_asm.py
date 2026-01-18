import contextlib
import functools
from llvmlite.ir import instructions, types, values
def asm(self, ftype, asm, constraint, args, side_effect, name=''):
    """
        Inline assembler.
        """
    asm = instructions.InlineAsm(ftype, asm, constraint, side_effect)
    return self.call(asm, args, name)