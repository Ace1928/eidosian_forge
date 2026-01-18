from itertools import combinations_with_replacement, product
from textwrap import dedent
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable
def mulpow(self):
    name = 'monomial_mulpow'
    template = dedent('        def %(name)s(A, B, k):\n            (%(A)s,) = A\n            (%(B)s,) = B\n            return (%(ABk)s,)\n        ')
    A = self._vars('a')
    B = self._vars('b')
    ABk = ['%s + %s*k' % (a, b) for a, b in zip(A, B)]
    code = template % {'name': name, 'A': ', '.join(A), 'B': ', '.join(B), 'ABk': ', '.join(ABk)}
    return self._build(code, name)