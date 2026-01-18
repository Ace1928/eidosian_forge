from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def pinv_with_derivatives(arr, input_type, derivatives, rcond):
    """
    Defines the matrix pseudo-inverse and its derivatives.

    Works with real or complex matrices.

    See the definition of func_with_deriv_to_uncert_func() for its
    detailed semantics.
    """
    inverse = numpy.linalg.pinv(arr, rcond)
    if issubclass(input_type, numpy.matrix):
        inverse = inverse.view(numpy.matrix)
    yield inverse
    inverse_mat = numpy.asmatrix(inverse)
    PA = arr * inverse_mat
    AP = inverse_mat * arr
    factor21 = inverse_mat * inverse_mat.H
    factor22 = numpy.eye(arr.shape[0]) - PA
    factor31 = numpy.eye(arr.shape[1]) - AP
    factor32 = inverse_mat.H * inverse_mat
    for derivative in derivatives:
        derivative_mat = numpy.asmatrix(derivative)
        term1 = -inverse_mat * derivative_mat * inverse_mat
        derivative_mat_H = derivative_mat.H
        term2 = factor21 * derivative_mat_H * factor22
        term3 = factor31 * derivative_mat_H * factor32
        yield (term1 + term2 + term3)