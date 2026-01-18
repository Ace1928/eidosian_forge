import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_heterogeneous_LiteralStrKeyDict_literal_values(self):
    """Check the literal values for a LiteralStrKeyDict requiring
        optimizations because it is heterogeneous.
        """

    def bar(d):
        ...

    @overload(bar)
    def ol_bar(d):
        a = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 'a'}

        def specific_ty(z):
            return types.literal(z) if types.maybe_literal(z) else typeof(z)
        expected = {types.literal(x): specific_ty(y) for x, y in a.items()}
        self.assertTrue(isinstance(d, types.LiteralStrKeyDict))
        self.assertEqual(d.literal_value, expected)
        self.assertEqual(hasattr(d, 'initial_value'), False)
        return lambda d: d

    @njit
    def foo():
        d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 'a'}
        bar(d)
    foo()