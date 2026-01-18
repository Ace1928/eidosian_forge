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
def contraction(a, b):
    """
    Calculates contraction of Fermionic operators a and b.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import F, Fd, contraction
    >>> p, q = symbols('p,q')
    >>> a, b = symbols('a,b', above_fermi=True)
    >>> i, j = symbols('i,j', below_fermi=True)

    A contraction is non-zero only if a quasi-creator is to the right of a
    quasi-annihilator:

    >>> contraction(F(a),Fd(b))
    KroneckerDelta(a, b)
    >>> contraction(Fd(i),F(j))
    KroneckerDelta(i, j)

    For general indices a non-zero result restricts the indices to below/above
    the fermi surface:

    >>> contraction(Fd(p),F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q)
    >>> contraction(F(p),Fd(q))
    KroneckerDelta(_a, q)*KroneckerDelta(p, q)

    Two creators or two annihilators always vanishes:

    >>> contraction(F(p),F(q))
    0
    >>> contraction(Fd(p),Fd(q))
    0

    """
    if isinstance(b, FermionicOperator) and isinstance(a, FermionicOperator):
        if isinstance(a, AnnihilateFermion) and isinstance(b, CreateFermion):
            if b.state.assumptions0.get('below_fermi'):
                return S.Zero
            if a.state.assumptions0.get('below_fermi'):
                return S.Zero
            if b.state.assumptions0.get('above_fermi'):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get('above_fermi'):
                return KroneckerDelta(a.state, b.state)
            return KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('a', above_fermi=True))
        if isinstance(b, AnnihilateFermion) and isinstance(a, CreateFermion):
            if b.state.assumptions0.get('above_fermi'):
                return S.Zero
            if a.state.assumptions0.get('above_fermi'):
                return S.Zero
            if b.state.assumptions0.get('below_fermi'):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get('below_fermi'):
                return KroneckerDelta(a.state, b.state)
            return KroneckerDelta(a.state, b.state) * KroneckerDelta(b.state, Dummy('i', below_fermi=True))
        return S.Zero
    else:
        t = (isinstance(i, FermionicOperator) for i in (a, b))
        raise ContractionAppliesOnlyToFermions(*t)