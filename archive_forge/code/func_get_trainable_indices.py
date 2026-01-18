import functools
from collections.abc import Sequence
import autoray as ar
import numpy as onp
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from numpy import ndarray
from . import single_dispatch  # pylint:disable=unused-import
from .utils import cast, cast_like, get_interface, requires_grad
@multi_dispatch(argnum=[0], tensor_list=[0])
def get_trainable_indices(values, like=None):
    """Returns a set containing the trainable indices of a sequence of
    values.

    Args:
        values (Iterable[tensor_like]): Sequence of tensor-like objects to inspect

    Returns:
        set[int]: Set containing the indices of the trainable tensor-like objects
        within the input sequence.

    **Example**

    >>> def cost_fn(params):
    ...     print("Trainable:", qml.math.get_trainable_indices(params))
    ...     return np.sum(np.sin(params[0] * params[1]))
    >>> values = [np.array([0.1, 0.2], requires_grad=True),
    ... np.array([0.5, 0.2], requires_grad=False)]
    >>> cost_fn(values)
    Trainable: {0}
    tensor(0.0899685, requires_grad=True)
    """
    trainable_params = set()
    for idx, p in enumerate(values):
        if requires_grad(p, interface=like):
            trainable_params.add(idx)
    return trainable_params