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
def is_q_annihilator(self):
    """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_annihilator
        0
        >>> Fd(i).is_q_annihilator
        -1
        >>> Fd(p).is_q_annihilator
        -1

        """
    if self.is_below_fermi:
        return -1
    return 0