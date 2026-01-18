import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def in_backprop(tensor, interface=None):
    """Returns True if the tensor is considered to be in a backpropagation environment, it works for Autograd,
    TensorFlow and Jax. It is not only checking the differentiability of the tensor like :func:`~.requires_grad`, but
    rather checking if the gradient is actually calculated.

    Args:
        tensor (tensor_like): input tensor
        interface (str): The name of the interface. Will be determined automatically
            if not provided.

    **Example**

    >>> x = tf.Variable([0.6, 0.1])
    >>> requires_grad(x)
    False
    >>> with tf.GradientTape() as tape:
    ...     print(requires_grad(x))
    True

    .. seealso:: :func:`~.requires_grad`
    """
    interface = interface or get_interface(tensor)
    if interface == 'tensorflow':
        import tensorflow as tf
        should_record_backprop = import_should_record_backprop()
        return should_record_backprop([tf.convert_to_tensor(tensor)])
    if interface == 'autograd':
        return isinstance(tensor, ArrayBox)
    if interface == 'jax':
        import jax
        return isinstance(tensor, jax.core.Tracer)
    if interface in {'numpy', 'scipy'}:
        return False
    raise ValueError(f'Cannot determine if {tensor} is in backpropagation.')