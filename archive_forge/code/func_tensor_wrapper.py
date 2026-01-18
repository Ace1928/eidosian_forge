from collections.abc import Sequence
import functools
from autograd import numpy as _np
from .tensor import tensor
def tensor_wrapper(obj):
    """Decorator that wraps callable objects and classes so that they both accept
    a ``requires_grad`` keyword argument, as well as returning a PennyLane
    :class:`~.tensor`.

    Only if the decorated object returns an ``ndarray`` is the
    output converted to a :class:`~.tensor`; this avoids superfluous conversion
    of scalars and other native-Python types.

    .. note::

        This wrapper does *not* enable autodifferentiation of the wrapped function,
        it merely adds support for :class:`~pennylane.numpy.tensor` output.

    Args:
        obj: a callable object or class

    **Example**

    By default, the ``ones`` function provided by Autograd
    constructs standard ``ndarray`` objects, and does not
    permit a ``requires_grad`` argument:

    >>> from autograd.numpy import ones
    >>> ones([2, 2])
    array([[1., 1.],
        [1., 1.]])
    >>> ones([2, 2], requires_grad=True)
    TypeError: ones() got an unexpected keyword argument 'requires_grad'

    ``tensor_wrapper`` both enables construction of :class:`~pennylane.numpy.tensor`
    objects, while also converting the output.

    >>> from pennylane import numpy as np
    >>> ones = np.tensor_wrapper(ones)
    >>> ones([2, 2], requires_grad=True)
    tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
    """

    @functools.wraps(obj)
    def _wrapped(*args, **kwargs):
        """Wrapped NumPy function"""
        tensor_kwargs = {}
        if 'requires_grad' in kwargs:
            tensor_kwargs['requires_grad'] = kwargs.pop('requires_grad')
        else:
            tensor_args = list(extract_tensors(args))
            if tensor_args:
                tensor_kwargs['requires_grad'] = _np.any([i.requires_grad for i in tensor_args])
        res = obj(*args, **kwargs)
        if isinstance(res, _np.ndarray):
            res = tensor(res, **tensor_kwargs)
        return res
    return _wrapped