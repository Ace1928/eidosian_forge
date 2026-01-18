import numpy as np
def mpmath_wrightomega(x):
    return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))