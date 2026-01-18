from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def check_no_repeated_axes(axes):
    axes_set = set(axes)
    if len(axes) != len(axes_set):
        raise NotImplementedError('FFT gradient for repeated axes not implemented.')