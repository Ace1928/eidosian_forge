import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def extrema(input, labels=None, index=None):
    """Calculate the minimums and maximums of the values of an array at labels,
    along with their positions.

    Args:
        input (cupy.ndarray): N-D image data to process.
        labels (cupy.ndarray, optional): Labels of features in input. If not
            None, must be same shape as `input`.
        index (int or sequence of ints, optional): Labels to include in output.
            If None (default), all values where non-zero `labels` are used.

    Returns:
        A tuple that contains the following values.

        **minimums (cupy.ndarray)**: Values of minimums in each feature.

        **maximums (cupy.ndarray)**: Values of maximums in each feature.

        **min_positions (tuple or list of tuples)**: Each tuple gives the N-D
        coordinates of the corresponding minimum.

        **max_positions (tuple or list of tuples)**: Each tuple gives the N-D
        coordinates of the corresponding maximum.

    .. seealso:: :func:`scipy.ndimage.extrema`
    """
    dims = numpy.array(input.shape)
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
    minimums, min_positions, maximums, max_positions = _select(input, labels, index, find_min=True, find_max=True, find_min_positions=True, find_max_positions=True)
    if min_positions.ndim == 0:
        min_positions = min_positions.item()
        max_positions = max_positions.item()
        return (minimums, maximums, tuple(min_positions // dim_prod % dims), tuple(max_positions // dim_prod % dims))
    min_positions = cupy.asnumpy(min_positions)
    max_positions = cupy.asnumpy(max_positions)
    min_positions = [tuple(v) for v in min_positions.reshape(-1, 1) // dim_prod % dims]
    max_positions = [tuple(v) for v in max_positions.reshape(-1, 1) // dim_prod % dims]
    return (minimums, maximums, min_positions, max_positions)