from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings
def test_MarginalDistribution():
    a1, p1, p2 = symbols('a1 p1 p2', positive=True)
    C = Multinomial('C', 2, p1, p2)
    B = MultivariateBeta('B', a1, C[0])
    MGR = MarginalDistribution(B, (C[0],))
    mgrc = Mul(Symbol('B'), Piecewise(ExprCondPair(Mul(Integer(2), Pow(Symbol('p1', positive=True), Indexed(IndexedBase(Symbol('C')), Integer(0))), Pow(Symbol('p2', positive=True), Indexed(IndexedBase(Symbol('C')), Integer(1))), Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)), Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(-1))), Eq(Add(Indexed(IndexedBase(Symbol('C')), Integer(0)), Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(2))), ExprCondPair(Integer(0), True)), Pow(gamma(Symbol('a1', positive=True)), Integer(-1)), gamma(Add(Symbol('a1', positive=True), Indexed(IndexedBase(Symbol('C')), Integer(0)))), Pow(gamma(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)), Pow(Indexed(IndexedBase(Symbol('B')), Integer(0)), Add(Symbol('a1', positive=True), Integer(-1))), Pow(Indexed(IndexedBase(Symbol('B')), Integer(1)), Add(Indexed(IndexedBase(Symbol('C')), Integer(0)), Integer(-1))))
    assert MGR(C) == mgrc