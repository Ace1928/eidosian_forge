from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def evaluate_deltas(e):
    """
    We evaluate KroneckerDelta symbols in the expression assuming Einstein summation.

    Explanation
    ===========

    If one index is repeated it is summed over and in effect substituted with
    the other one. If both indices are repeated we substitute according to what
    is the preferred index.  this is determined by
    KroneckerDelta.preferred_index and KroneckerDelta.killable_index.

    In case there are no possible substitutions or if a substitution would
    imply a loss of information, nothing is done.

    In case an index appears in more than one KroneckerDelta, the resulting
    substitution depends on the order of the factors.  Since the ordering is platform
    dependent, the literal expression resulting from this function may be hard to
    predict.

    Examples
    ========

    We assume the following:

    >>> from sympy import symbols, Function, Dummy, KroneckerDelta
    >>> from sympy.physics.secondquant import evaluate_deltas
    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)
    >>> a,b = symbols('a b', above_fermi=True, cls=Dummy)
    >>> p,q = symbols('p q', cls=Dummy)
    >>> f = Function('f')
    >>> t = Function('t')

    The order of preference for these indices according to KroneckerDelta is
    (a, b, i, j, p, q).

    Trivial cases:

    >>> evaluate_deltas(KroneckerDelta(i,j)*f(i))       # d_ij f(i) -> f(j)
    f(_j)
    >>> evaluate_deltas(KroneckerDelta(i,j)*f(j))       # d_ij f(j) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(p))       # d_ip f(p) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(p))       # d_qp f(p) -> f(q)
    f(_q)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(q))       # d_qp f(q) -> f(p)
    f(_p)

    More interesting cases:

    >>> evaluate_deltas(KroneckerDelta(i,p)*t(a,i)*f(p,q))
    f(_i, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(a,p)*t(a,i)*f(p,q))
    f(_a, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(p,q)*f(p,q))
    f(_p, _p)

    Finally, here are some cases where nothing is done, because that would
    imply a loss of information:

    >>> evaluate_deltas(KroneckerDelta(i,p)*f(q))
    f(_q)*KroneckerDelta(_i, _p)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(i))
    f(_i)*KroneckerDelta(_i, _p)
    """
    accepted_functions = (Add,)
    if isinstance(e, accepted_functions):
        return e.func(*[evaluate_deltas(arg) for arg in e.args])
    elif isinstance(e, Mul):
        deltas = []
        indices = {}
        for i in e.args:
            for s in i.free_symbols:
                if s in indices:
                    indices[s] += 1
                else:
                    indices[s] = 0
            if isinstance(i, KroneckerDelta):
                deltas.append(i)
        for d in deltas:
            if d.killable_index.is_Symbol and indices[d.killable_index]:
                e = e.subs(d.killable_index, d.preferred_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            elif d.preferred_index.is_Symbol and indices[d.preferred_index] and d.indices_contain_equal_information:
                e = e.subs(d.preferred_index, d.killable_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            else:
                pass
        return e
    else:
        return e