from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, \
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution
@singledispatch
def do_sample_pymc(dist):
    return None