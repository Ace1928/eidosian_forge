import contextlib
import functools
from llvmlite.ir import instructions, types, values
def atomic_rmw(self, op, ptr, val, ordering, name=''):
    inst = instructions.AtomicRMW(self.block, op, ptr, val, ordering, name=name)
    self._insert(inst)
    return inst