from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants
def equivalence_hypergeometric(A, B, func):
    x = func.args[0]
    I1 = factor(cancel(A.diff(x) / 2 + A ** 2 / 4 - B))
    J1 = factor(cancel(x ** 2 * I1 + S(1) / 4))
    num, dem = J1.as_numer_denom()
    num = powdenest(expand(num))
    dem = powdenest(expand(dem))

    def _power_counting(num):
        _pow = {0}
        for val in num:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    _pow.add(val.as_base_exp()[1])
                elif val == x:
                    _pow.add(val.as_base_exp()[1])
                else:
                    _pow.update(_power_counting(val.args))
        return _pow
    pow_num = _power_counting((num,))
    pow_dem = _power_counting((dem,))
    pow_dem.update(pow_num)
    _pow = pow_dem
    k = gcd(_pow)
    I0 = powdenest(simplify(factor((J1 / k ** 2 - S(1) / 4) / (x ** k) ** 2)), force=True)
    I0 = factor(cancel(powdenest(I0.subs(x, x ** (S(1) / k)), force=True)))
    if not I0.is_rational_function(x):
        return None
    num, dem = I0.as_numer_denom()
    max_num_pow = max(_power_counting((num,)))
    dem_args = dem.args
    sing_point = []
    dem_pow = []
    for arg in dem_args:
        if arg.has(x):
            if isinstance(arg, Pow):
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg.as_base_exp()[0], x).keys())[0])
            else:
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg, x).keys())[0])
    dem_pow.sort()
    if equivalence(max_num_pow, dem_pow) == '2F1':
        return {'I0': I0, 'k': k, 'sing_point': sing_point, 'type': '2F1'}
    else:
        return None