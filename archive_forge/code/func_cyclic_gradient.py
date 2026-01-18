from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def cyclic_gradient(data: np.ndarray, *, edge_order: Literal[1, 2]=1, axis: int=-1) -> np.ndarray:
    """Estimate the gradient of a function over a uniformly sampled,
    periodic domain.

    This is essentially the same as `np.gradient`, except that edge effects
    are handled by wrapping the observations (i.e. assuming periodicity)
    rather than extrapolation.

    Parameters
    ----------
    data : np.ndarray
        The function values observed at uniformly spaced positions on
        a periodic domain
    edge_order : {1, 2}
        The order of the difference approximation used for estimating
        the gradient
    axis : int
        The axis along which gradients are calculated.

    Returns
    -------
    grad : np.ndarray like ``data``
        The gradient of ``data`` taken along the specified axis.

    See Also
    --------
    numpy.gradient

    Examples
    --------
    This example estimates the gradient of cosine (-sine) from 64
    samples using direct (aperiodic) and periodic gradient
    calculation.

    >>> import matplotlib.pyplot as plt
    >>> x = 2 * np.pi * np.linspace(0, 1, num=64, endpoint=False)
    >>> y = np.cos(x)
    >>> grad = np.gradient(y)
    >>> cyclic_grad = librosa.util.cyclic_gradient(y)
    >>> true_grad = -np.sin(x) * 2 * np.pi / len(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, true_grad, label='True gradient', linewidth=5,
    ...          alpha=0.35)
    >>> ax.plot(x, cyclic_grad, label='cyclic_gradient')
    >>> ax.plot(x, grad, label='np.gradient', linestyle=':')
    >>> ax.legend()
    >>> # Zoom into the first part of the sequence
    >>> ax.set(xlim=[0, np.pi/16], ylim=[-0.025, 0.025])
    """
    padding = [(0, 0)] * data.ndim
    padding[axis] = (edge_order, edge_order)
    data_pad = np.pad(data, padding, mode='wrap')
    grad = np.gradient(data_pad, edge_order=edge_order, axis=axis)
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(edge_order, -edge_order)
    grad_slice: np.ndarray = grad[tuple(slices)]
    return grad_slice