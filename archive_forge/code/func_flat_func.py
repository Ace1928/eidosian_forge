from __future__ import absolute_import
from builtins import range
import scipy.integrate
import autograd.numpy as np
from autograd.extend import primitive, defvjp_argnums
from autograd import make_vjp
from autograd.misc import flatten
from autograd.builtins import tuple
def flat_func(y, t, flat_args):
    return func(y, t, *unflatten(flat_args))