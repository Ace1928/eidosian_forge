from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
def breakup(eq):
    """break up powers of eq when treated as a Mul:
                   b**(Rational*e) -> b**e, Rational
                commutatives come back as a dictionary {b**e: Rational}
                noncommutatives come back as a list [(b**e, Rational)]
            """
    c, nc = (defaultdict(int), [])
    for a in Mul.make_args(eq):
        a = powdenest(a)
        b, e = base_exp(a)
        if e is not S.One:
            co, _ = e.as_coeff_mul()
            b = Pow(b, e / co)
            e = co
        if a.is_commutative:
            c[b] += e
        else:
            nc.append([b, e])
    return (c, nc)