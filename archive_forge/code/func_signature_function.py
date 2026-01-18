import spherogram
import snappy
import numpy as np
import mpmath
from sage.all import PolynomialRing, LaurentPolynomialRing, RR, ZZ, RealField, ComplexField, matrix, arccos, exp
def signature_function(L, prec=53):
    """
    Computes the signature function sigma of K via numerical methods.
    Returns two lists, the first representing a partition of [0, 1]:

         x_0 = 0 < x_1 < x_2 < ... < x_n = 1

    and the second list consisting of the values [v_0, ... , v_(n-1)]
    of sigma on the interval (x_i, x_(i+1)).  Currently, the value of
    sigma *at* x_i is not computed.
    """
    V = L.seifert_matrix()
    return signature_function_of_integral_matrix(V)