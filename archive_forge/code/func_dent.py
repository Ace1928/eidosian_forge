import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def dent(individual, lambda_=0.85):
    """Test problem Dent. Two-objective problem with a "dent". *individual* has
    two attributes that take values in [-1.5, 1.5].
    From: Schuetze, O., Laumanns, M., Tantar, E., Coello Coello, C.A., & Talbi, E.-G. (2010).
    Computing gap free Pareto front approximations with stochastic search algorithms.
    Evolutionary Computation, 18(1), 65--96. doi:10.1162/evco.2010.18.1.18103

    Note that in that paper Dent source is stated as:
    K. Witting and M. Hessel von Molo. Private communication, 2006.
    """
    d = lambda_ * exp(-(individual[0] - individual[1]) ** 2)
    f1 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + sqrt(1 + (individual[0] - individual[1]) ** 2) + individual[0] - individual[1]) + d
    f2 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + sqrt(1 + (individual[0] - individual[1]) ** 2) - individual[0] + individual[1]) + d
    return (f1, f2)