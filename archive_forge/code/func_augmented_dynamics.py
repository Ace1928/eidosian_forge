from __future__ import absolute_import
from builtins import range
import scipy.integrate
import autograd.numpy as np
from autograd.extend import primitive, defvjp_argnums
from autograd import make_vjp
from autograd.misc import flatten
from autograd.builtins import tuple
def augmented_dynamics(augmented_state, t, flat_args):
    y, vjp_y, _, _ = unpack(augmented_state)
    vjp_all, dy_dt = make_vjp(flat_func, argnum=(0, 1, 2))(y, t, flat_args)
    vjp_y, vjp_t, vjp_args = vjp_all(-vjp_y)
    return np.hstack((dy_dt, vjp_y, vjp_t, vjp_args))