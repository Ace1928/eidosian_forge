import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def cast_like(tensor1, tensor2):
    """Casts a tensor to the same dtype as another.

    Args:
        tensor1 (tensor_like): tensor to cast
        tensor2 (tensor_like): tensor with corresponding dtype to cast to

    Returns:
        tensor_like: a tensor with the same shape and values as ``tensor1`` and the
        same dtype as ``tensor2``

    **Example**

    >>> x = torch.tensor([1, 2])
    >>> y = torch.tensor([3., 4.])
    >>> cast_like(x, y)
    tensor([1., 2.])
    """
    if isinstance(tensor2, tuple) and len(tensor2) > 0:
        tensor2 = tensor2[0]
    if isinstance(tensor2, ArrayBox):
        dtype = ar.to_numpy(tensor2._value).dtype.type
    elif not is_abstract(tensor2):
        dtype = ar.to_numpy(tensor2).dtype.type
    else:
        dtype = tensor2.dtype
    return cast(tensor1, dtype)