import numpy as np
def lambertw_pade():
    derivs = [mpmath.diff(mpmath.lambertw, 0, n=n) for n in range(6)]
    p, q = mpmath.pade(derivs, 3, 2)
    return (p, q)