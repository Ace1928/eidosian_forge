from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod
import itertools
def test_Range_eval_imageset():
    a, b, c = symbols('a b c')
    assert imageset(x, a * (x + b) + c, Range(3)) == imageset(x, a * x + a * b + c, Range(3))
    eq = (x + 1) ** 2
    assert imageset(x, eq, Range(3)).lamda.expr == eq
    eq = a * (x + b) + c
    r = Range(3, -3, -2)
    imset = imageset(x, eq, r)
    assert imset.lamda.expr != eq
    assert list(imset) == [eq.subs(x, i).expand() for i in list(r)]