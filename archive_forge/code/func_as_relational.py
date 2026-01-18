from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda, BadSignatureError
from sympy.core.logic import fuzzy_bool
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import And, as_Boolean
from sympy.utilities.iterables import sift, flatten, has_dups
from sympy.utilities.exceptions import sympy_deprecation_warning
from .contains import Contains
from .sets import Set, Union, FiniteSet, SetKind
def as_relational(self, other):
    f = Lambda(self.sym, self.condition)
    if isinstance(self.sym, Tuple):
        f = f(*other)
    else:
        f = f(other)
    return And(f, self.base_set.contains(other))