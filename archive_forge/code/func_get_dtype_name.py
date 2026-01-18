import autoray as ar
from .is_independent import is_independent
from .matrix_manipulation import expand_matrix, reduce_matrices, get_batch_size
from .multi_dispatch import (
from .quantum import (
from .fidelity import fidelity, fidelity_statevector
from .utils import (
def get_dtype_name(x) -> str:
    """An interface independent way of getting the name of the datatype.

    >>> x = tf.Variable(0.1)
    >>> qml.math.get_dtype_name(tf.Variable(0.1))
    'float32'
    """
    return ar.get_dtype_name(x)