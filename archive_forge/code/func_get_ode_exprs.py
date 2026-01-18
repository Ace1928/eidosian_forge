from __future__ import division, print_function, absolute_import
import math
import numpy as np
from ..util import import_
from ..core import RecoverableError
from ..symbolic import ScaledSys
def get_ode_exprs(logc=False, logt=False, reduced=0, base2=False):
    """
    reduced:
    0: A, B, C
    1: B, C
    2: A, C
    3: A, B
    """
    if base2:
        lnb = math.log(2)
        raise NotImplementedError('TODO')
    else:
        lnb = 1
    if reduced not in (0, 1, 2, 3):
        raise NotImplementedError('What invariant did you have in mind?')

    def dydt(x, y, p, backend=np):
        if backend == np and (not logc):
            if np.any(np.asarray(y) < 0):
                raise RecoverableError
        exp = backend.exp
        k1, k2, k3 = p[:3]
        if reduced:
            A0, B0, C0 = p[3:]
        if logc:
            if reduced == 0:
                expy = A, B, C = list(map(exp, y))
            elif reduced == 1:
                expy = B, C = list(map(exp, y))
            elif reduced == 2:
                expy = A, C = list(map(exp, y))
            elif reduced == 3:
                expy = A, B = list(map(exp, y))
        elif reduced == 0:
            A, B, C = y
        elif reduced == 1:
            B, C = y
        elif reduced == 2:
            A, C = y
        elif reduced == 3:
            A, B = y
        if reduced == 1:
            A = A0 + B0 + C0 - B - C
        elif reduced == 2:
            B = A0 + B0 + C0 - A - C
        elif reduced == 3:
            C = A0 + B0 + C0 - A - B
        r1 = k1 * A
        r2 = k2 * B * C
        r3 = k3 * B * B
        f = [r2 - r1, r1 - r2 - r3, r3]
        if reduced == 0:
            pass
        elif reduced == 1:
            f = [f[1], f[2]]
        elif reduced == 2:
            f = [f[0], f[2]]
        elif reduced == 3:
            f = [f[0], f[1]]
        if logc:
            f = [f_ / ey for ey, f_ in zip(expy, f)]
        if logt:
            ex = exp(x)
            f = [ex * f_ for f_ in f]
        return f

    def jac(x, y, p, backend=np):
        exp = backend.exp
        k1, k2, k3 = p[:3]
        if reduced:
            A0, B0, C0 = p[3:]
            I0 = A0 + B0 + C0
        if logc:
            if reduced == 0:
                A, B, C = list(map(exp, y))
            elif reduced == 1:
                B, C = list(map(exp, y))
                A = I0 - B - C
            elif reduced == 2:
                A, C = list(map(exp, y))
                B = I0 - A - C
            elif reduced == 3:
                A, B = list(map(exp, y))
                C = I0 - A - B
        elif reduced == 0:
            A, B, C = y
        elif reduced == 1:
            B, C = y
            A = I0 - B - C
        elif reduced == 2:
            A, C = y
            B = I0 - A - C
        elif reduced == 3:
            A, B = y
            C = I0 - A - B
        liny = (A, B, C)
        r1 = k1 * A
        r2 = k2 * B * C
        r3 = k3 * B * B
        f = [r2 - r1, r1 - r2 - r3, r3]
        dr = [[k1, 0, 0], [0, k2 * C, k2 * B], [0, 2 * k3 * B, 0]]
        if reduced == 1:
            dr[0] = [0, -k1, -k1]
        elif reduced == 2:
            dr[1] = [-k2 * C, 0, k2 * (I0 - 2 * C - A)]
            dr[2] = [-2 * k3 * (I0 - A - C), 0, -2 * k3 * (I0 - A - C)]
        elif reduced == 3:
            dr[1] = [-k2 * B, k2 * (I0 - A) - 2 * k2 * B, 0]

        def _jfct(ri, ci):
            if logc:
                return liny[ci] / liny[ri]
            else:
                return 1

        def _jtrm(ri, ji):
            if logc and ri == ji:
                return -f[ri] / liny[ri]
            else:
                return 0

        def _o(expr):
            if logt:
                return exp(x) * lnb * expr
            else:
                return expr
        j1 = [_o(_jtrm(0, i) + _jfct(0, i) * (dr[1][i] - dr[0][i])) for i in range(3) if i != reduced - 1]
        j2 = [_o(_jtrm(1, i) + _jfct(1, i) * (dr[0][i] - dr[1][i] - dr[2][i])) for i in range(3) if i != reduced - 1]
        j3 = [_o(_jtrm(2, i) + _jfct(2, i) * dr[2][i]) for i in range(3) if i != reduced - 1]
        return [j for i, j in enumerate([j1, j2, j3]) if i != reduced - 1]
    return (dydt, jac)