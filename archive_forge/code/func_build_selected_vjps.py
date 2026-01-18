from __future__ import absolute_import
from builtins import range
import scipy.integrate
import autograd.numpy as np
from autograd.extend import primitive, defvjp_argnums
from autograd import make_vjp
from autograd.misc import flatten
from autograd.builtins import tuple
def build_selected_vjps(argnums, ans, combined_args, kwargs):
    vjp_func = all_vjp_builder(ans, *combined_args, **kwargs)

    def chosen_vjps(g):
        all_vjps = vjp_func(g)
        return [all_vjps[argnum] for argnum in argnums]
    return chosen_vjps