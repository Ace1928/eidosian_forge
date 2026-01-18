from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def check_even_shape(shape):
    if shape[-1] % 2 != 0:
        raise NotImplementedError('Real FFT gradient for odd lengthed last axes is not implemented.')