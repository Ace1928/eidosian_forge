import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
@classmethod
def irregular(cls, ntop, *matrices, **kwargs):
    """Return a matrix filled by the given matrices which
      are listed in order of appearance from left to right, top to
      bottom as they first appear in the matrix. They must fill the
      matrix completely.

      Examples
      ========

      >>> from sympy import ones, Matrix
      >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
      ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)
      Matrix([
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [4, 2, 2, 2, 5, 5],
        [6, 6, 7, 7, 5, 5]])
      """
    ntop = as_int(ntop)
    b = [i.as_explicit() if hasattr(i, 'as_explicit') else i for i in matrices]
    q = list(range(len(b)))
    dat = [i.rows for i in b]
    active = [q.pop(0) for _ in range(ntop)]
    cols = sum([b[i].cols for i in active])
    rows = []
    while any(dat):
        r = []
        for a, j in enumerate(active):
            r.extend(b[j][-dat[j], :])
            dat[j] -= 1
            if dat[j] == 0 and q:
                active[a] = q.pop(0)
        if len(r) != cols:
            raise ValueError(filldedent('\n                Matrices provided do not appear to fill\n                the space completely.'))
        rows.append(r)
    return cls._new(rows)