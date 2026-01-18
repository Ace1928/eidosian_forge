from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def get_fft2_args(a, s=None, axes=(-2, -1), norm=None, *args, **kwargs):
    return (axes, s, norm)