from __future__ import absolute_import
from builtins import range
import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps
@unflatten_optimizer
def rmsprop(grad, x, callback=None, num_iters=100, step_size=0.1, gamma=0.9, eps=10 ** (-8)):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback:
            callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g ** 2 * (1 - gamma)
        x = x - step_size * g / (np.sqrt(avg_sq_grad) + eps)
    return x