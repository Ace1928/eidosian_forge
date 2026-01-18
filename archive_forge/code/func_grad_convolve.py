from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def grad_convolve(argnum, ans, A, B, axes=None, dot_axes=[(), ()], mode='full'):
    assert mode in ['valid', 'full'], 'Grad for mode {0} not yet implemented'.format(mode)
    axes, shapes = parse_axes(A.shape, B.shape, axes, dot_axes, mode)
    if argnum == 0:
        X, Y = (A, B)
        _X_, _Y_ = ('A', 'B')
        ignore_Y = 'ignore_B'
    elif argnum == 1:
        X, Y = (B, A)
        _X_, _Y_ = ('B', 'A')
        ignore_Y = 'ignore_A'
    else:
        raise NotImplementedError("Can't take grad of convolve w.r.t. arg {0}".format(argnum))
    if mode == 'full':
        new_mode = 'valid'
    elif any([x_size > y_size for x_size, y_size in zip(shapes[_X_]['conv'], shapes[_Y_]['conv'])]):
        new_mode = 'full'
    else:
        new_mode = 'valid'

    def vjp(g):
        result = convolve(g, Y[flipped_idxs(Y.ndim, axes[_Y_]['conv'])], axes=[axes['out']['conv'], axes[_Y_]['conv']], dot_axes=[axes['out'][ignore_Y], axes[_Y_]['ignore']], mode=new_mode)
        new_order = npo.argsort(axes[_X_]['ignore'] + axes[_X_]['dot'] + axes[_X_]['conv'])
        return np.transpose(result, new_order)
    return vjp