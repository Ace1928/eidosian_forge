from sympy.core.relational import Eq, is_eq
from sympy.core.basic import Basic
from sympy.core.logic import fuzzy_and, fuzzy_bool
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.sets.sets import tfn, ProductSet, Interval, FiniteSet, Set
def all_in_both():
    s_set = set(lhs.args)
    o_set = set(rhs.args)
    yield fuzzy_and((lhs._contains(e) for e in o_set - s_set))
    yield fuzzy_and((rhs._contains(e) for e in s_set - o_set))