import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
@multi_dispatch(argnum=[0, 2])
def scatter_element_add(tensor, index, value, like=None):
    """In-place addition of a multidimensional value over various
    indices of a tensor.

    Args:
        tensor (tensor_like[float]): Tensor to add the value to
        index (tuple or list[tuple]): Indices to which to add the value
        value (float or tensor_like[float]): Value to add to ``tensor``
        like (str): Manually chosen interface to dispatch to.
    Returns:
        tensor_like[float]: The tensor with the value added at the given indices.

    **Example**

    >>> tensor = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> index = (1, 2)
    >>> value = -3.1
    >>> qml.math.scatter_element_add(tensor, index, value)
    tensor([[ 0.1000,  0.2000,  0.3000],
            [ 0.4000,  0.5000, -2.5000]])

    If multiple indices are given, in the form of a list of tuples, the
    ``k`` th tuple is interpreted to contain the ``k`` th entry of all indices:

    >>> indices = [(1, 0), (2, 1)] # This will modify the entries (1, 2) and (0, 1)
    >>> values = torch.tensor([10, 20])
    >>> qml.math.scatter_element_add(tensor, indices, values)
    tensor([[ 0.1000, 20.2000,  0.3000],
            [ 0.4000,  0.5000, 10.6000]])
    """
    if len(np.shape(tensor)) == 0 and index == ():
        return tensor + value
    return np.scatter_element_add(tensor, index, value, like=like)