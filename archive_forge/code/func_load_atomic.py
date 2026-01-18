import contextlib
import functools
from llvmlite.ir import instructions, types, values
def load_atomic(self, ptr, ordering, align, name=''):
    """
        Load value from pointer, with optional guaranteed alignment:
            name = *ptr
        """
    if not isinstance(ptr.type, types.PointerType):
        msg = 'cannot load from value of type %s (%r): not a pointer'
        raise TypeError(msg % (ptr.type, str(ptr)))
    ld = instructions.LoadAtomicInstr(self.block, ptr, ordering, align, name)
    self._insert(ld)
    return ld