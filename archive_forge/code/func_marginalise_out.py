from math import prod
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module
def marginalise_out(self, expr, rv):
    from sympy.concrete.summations import Sum
    if isinstance(rv, RandomSymbol):
        dom = rv.pspace.set
    elif isinstance(rv, Indexed):
        dom = rv.base.component_domain(rv.pspace.component_domain(rv.args[1]))
    expr = expr.xreplace({rv: rv.pspace.symbol})
    if rv.pspace.is_Continuous:
        expr = Integral(expr, (rv.pspace.symbol, dom))
    elif rv.pspace.is_Discrete:
        if dom in (S.Integers, S.Naturals, S.Naturals0):
            dom = (dom.inf, dom.sup)
        expr = Sum(expr, (rv.pspace.symbol, dom))
    return expr