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
@property
def has_q_annihilators(self):
    """
        Return 0 if the rightmost argument of the first argument is a not a
        q_annihilator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_annihilators
        -1
        >>> NO(F(i)*F(a)).has_q_annihilators
        1
        >>> NO(Fd(a)*F(i)).has_q_annihilators
        0

        """
    return self.args[0].args[-1].is_q_annihilator