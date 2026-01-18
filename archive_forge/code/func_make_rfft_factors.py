from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def make_rfft_factors(axes, resshape, facshape, normshape, norm):
    """ make the compression factors and compute the normalization
        for irfft and rfft.
    """
    N = 1.0
    for n in normshape:
        N = N * n
    fac = anp.zeros(resshape)
    fac[...] = 2
    index = [slice(None)] * len(resshape)
    if facshape[-1] <= resshape[axes[-1]]:
        index[axes[-1]] = (0, facshape[-1] - 1)
    else:
        index[axes[-1]] = (0,)
    fac[tuple(index)] = 1
    if norm is None:
        fac /= N
    return fac