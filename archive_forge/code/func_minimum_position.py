import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def minimum_position(input, labels=None, index=None):
    """Find the positions of the minimums of the values of an array at labels.

    For each region specified by `labels`, the position of the minimum
    value of `input` within the region is returned.

    Args:
        input (cupy.ndarray):
            Array of values. For each region specified by `labels`, the
            minimal values of `input` over the region is computed.
        labels (cupy.ndarray, optional): An array of integers marking different
            regions over which the position of the minimum value of `input` is
            to be computed. `labels` must have the same shape as `input`. If
            `labels` is not specified, the location of the first minimum over
            the whole array is returned.

            The `labels` argument only works when `index` is specified.
        index (array_like, optional): A list of region labels that are taken
            into account for finding the location of the minima. If `index` is
            None, the ``first`` minimum over all elements where `labels` is
            non-zero is returned.

            The `index` argument only works when `labels` is specified.

    Returns:
        Tuple of ints or list of tuples of ints that specify the location of
        minima of `input` over the regions determined by `labels` and  whose
        index is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is returned
        specifying the location of the first minimal value of `input`.

    .. note::
        When `input` has multiple identical minima within a labeled region,
        the coordinates returned are not guaranteed to match those returned by
        SciPy.

    .. seealso:: :func:`scipy.ndimage.minimum_position`
    """
    dims = numpy.asarray(input.shape)
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
    result = _select(input, labels, index, find_min_positions=True)[0]
    if result.ndim == 0:
        result = int(result)
    else:
        result = cupy.asnumpy(result)
    if cupy.isscalar(result):
        return tuple(result // dim_prod % dims)
    return [tuple(v) for v in result.reshape(-1, 1) // dim_prod % dims]