import contextlib
import functools
from llvmlite.ir import instructions, types, values
def assume(self, cond):
    """
        Optimizer hint: assume *cond* is always true.
        """
    fn = self.module.declare_intrinsic('llvm.assume')
    return self.call(fn, [cond])