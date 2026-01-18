import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_uniop_intrinsic_float('llvm.convert.to.fp16')
def convert_to_fp16(self, a):
    """
        Convert the given FP number to an i16
        """