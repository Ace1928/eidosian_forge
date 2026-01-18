import contextlib
import functools
from llvmlite.ir import instructions, types, values
def cmpxchg(self, ptr, cmp, val, ordering, failordering=None, name=''):
    """
        Atomic compared-and-set:
            atomic {
                old = *ptr
                success = (old == cmp)
                if (success)
                    *ptr = val
                }
            name = { old, success }

        If failordering is `None`, the value of `ordering` is used.
        """
    failordering = ordering if failordering is None else failordering
    inst = instructions.CmpXchg(self.block, ptr, cmp, val, ordering, failordering, name=name)
    self._insert(inst)
    return inst