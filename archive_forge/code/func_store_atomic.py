import contextlib
import functools
from llvmlite.ir import instructions, types, values
def store_atomic(self, value, ptr, ordering, align):
    """
        Store value to pointer, with optional guaranteed alignment:
            *ptr = name
        """
    if not isinstance(ptr.type, types.PointerType):
        msg = 'cannot store to value of type %s (%r): not a pointer'
        raise TypeError(msg % (ptr.type, str(ptr)))
    if ptr.type.pointee != value.type:
        raise TypeError('cannot store %s to %s: mismatching types' % (value.type, ptr.type))
    st = instructions.StoreAtomicInstr(self.block, value, ptr, ordering, align)
    self._insert(st)
    return st