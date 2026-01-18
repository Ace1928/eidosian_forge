import contextlib
import functools
from llvmlite.ir import instructions, types, values
def store_reg(self, value, reg_type, reg_name, name=''):
    """
        Store an LLVM value inside a register
        Example:
          store_reg(Constant(IntType(32), 0xAAAAAAAA), IntType(32), "eax")
        """
    ftype = types.FunctionType(types.VoidType(), [reg_type])
    return self.asm(ftype, '', '{%s}' % reg_name, [value], True, name)