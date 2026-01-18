from cupy import _core
from cupy._creation import basic
from cupy.random import _distributions
from cupy.random import _generator
def random_integers(low, high=None, size=None):
    """Return a scalar or an array of integer values over ``[low, high]``

    Each element of returned values are independently sampled from
    uniform distribution over closed interval ``[low, high]``.

    Args:
        low (int): If ``high`` is not ``None``,
            it is the lower bound of the interval.
            Otherwise, it is the **upper** bound of the interval
            and the lower bound is set to ``1``.
        high (int): Upper bound of the interval.
        size (None or int or tuple of ints): The shape of returned value.

    Returns:
        int or cupy.ndarray of ints: If size is ``None``,
        it is single integer sampled.
        If size is integer, it is the 1D-array of length ``size`` element.
        Otherwise, it is the array whose shape specified by ``size``.
    """
    if high is None:
        high = low
        low = 1
    return randint(low, high + 1, size)