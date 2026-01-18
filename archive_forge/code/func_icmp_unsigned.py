import contextlib
import functools
from llvmlite.ir import instructions, types, values
def icmp_unsigned(self, cmpop, lhs, rhs, name=''):
    """
        Unsigned integer (or pointer) comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    return self._icmp('u', cmpop, lhs, rhs, name)