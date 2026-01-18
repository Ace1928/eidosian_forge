import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Args:
        input (cupy.ndarray): Data from which to calculate center-of-mass. The
            masses can either be positive or negative.
        labels (cupy.ndarray, optional): Labels for objects in `input`, as
            enerated by `ndimage.label`. Only used with `index`. Dimensions
            must be the same as `input`.
        index (int or sequence of ints, optional): Labels for which to
            calculate centers-of-mass. If not specified, all labels greater
            than zero are used. Only used with `labels`.

    Returns:
        tuple or list of tuples: Coordinates of centers-of-mass.

    .. seealso:: :func:`scipy.ndimage.center_of_mass`
    """
    normalizer = sum(input, labels, index)
    grids = cupy.ogrid[[slice(0, i) for i in input.shape]]
    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer for dir in range(input.ndim)]
    is_0dim_array = isinstance(results[0], cupy.ndarray) and results[0].ndim == 0
    if is_0dim_array:
        return tuple((res for res in results))
    return [v for v in cupy.stack(results, axis=-1)]