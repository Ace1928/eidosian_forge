from ..libmp.backend import xrange
from .calculus import defun
def init_levin(m):
    variant = kwargs.get('levin_variant', 'u')
    if isinstance(variant, str):
        if variant == 'all':
            variant = ['u', 'v', 't']
        else:
            variant = [variant]
    for s in variant:
        L = levin_class(method=m, variant=s)
        L.ctx = ctx
        L.name = m + '(' + s + ')'
        summer.append(L)