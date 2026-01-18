from typing import Any, Sequence, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.util import PublicAPI
@PublicAPI(stability='alpha')
def create_ragged_ndarray(values: Sequence[np.ndarray]) -> np.ndarray:
    """Create an array that contains arrays of different length

    If you're working with variable-length arrays like images, use this function to
    create ragged arrays instead of ``np.array``.

    .. note::
        ``np.array`` fails to construct ragged arrays if the input arrays have a uniform
        first dimension:

        .. testsetup::

            import numpy as np
            from ray.air.util.tensor_extensions.utils import create_ragged_ndarray

        .. doctest::

            >>> values = [np.zeros((3, 1)), np.zeros((3, 2))]
            >>> np.array(values, dtype=object)
            Traceback (most recent call last):
                ...
            ValueError: could not broadcast input array from shape (3,1) into shape (3,)
            >>> create_ragged_ndarray(values)
            array([array([[0.],
                          [0.],
                          [0.]]), array([[0., 0.],
                                         [0., 0.],
                                         [0., 0.]])], dtype=object)

        Or if you're creating a ragged array from a single array:

        .. doctest::

            >>> values = [np.zeros((3, 1))]
            >>> np.array(values, dtype=object)[0].dtype
            dtype('O')
            >>> create_ragged_ndarray(values)[0].dtype
            dtype('float64')

        ``create_ragged_ndarray`` avoids the limitations of ``np.array`` by creating an
        empty array and filling it with pointers to the variable-length arrays.
    """
    arr = np.empty(len(values), dtype=object)
    arr[:] = list(values)
    return arr