from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def rfft_grad(get_args, irfft_fun, ans, x, *args, **kwargs):
    axes, s, norm = get_args(x, *args, **kwargs)
    vs = vspace(x)
    gvs = vspace(ans)
    check_no_repeated_axes(axes)
    if s is None:
        s = [vs.shape[i] for i in axes]
    check_even_shape(s)
    gs = list(s)
    gs[-1] = gs[-1] // 2 + 1
    fac = make_rfft_factors(axes, gvs.shape, gs, s, norm)

    def vjp(g):
        g = anp.conj(g / fac)
        r = match_complex(x, truncate_pad(irfft_fun(g, *args, **kwargs), vs.shape))
        return r
    return vjp