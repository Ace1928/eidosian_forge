import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime
def histogramdd(sample, bins=10, range=None, weights=None, density=False):
    """Compute the multidimensional histogram of some data.

    Args:
        sample (cupy.ndarray): The data to be histogrammed. (N, D) or (D, N)
            array

            Note the unusual interpretation of sample when an array_like:

            * When an array, each row is a coordinate in a D-dimensional
              space - such as ``histogramdd(cupy.array([p1, p2, p3]))``.
            * When an array_like, each element is the list of values for single
              coordinate - such as ``histogramdd((X, Y, Z))``.

            The first form should be preferred.
        bins (int or tuple of int or cupy.ndarray): The bin specification:

            * A sequence of arrays describing the monotonically increasing bin
              edges along each dimension.
            * The number of bins for each dimension (nx, ny, ... =bins)
            * The number of bins for all dimensions (nx=ny=...=bins).
        range (sequence, optional): A sequence of length D, each an optional
            (lower, upper) tuple giving the outer bin edges to be used if the
            edges are not given explicitly in `bins`. An entry of None in the
            sequence results in the minimum and maximum values being used for
            the corresponding dimension. The default, None, is equivalent to
            passing a tuple of D None values.
        weights (cupy.ndarray): An array of values `w_i` weighing each sample
            `(x_i, y_i, z_i, ...)`. The values of the returned histogram are
            equal to the sum of the weights belonging to the samples falling
            into each bin.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.

    Returns:
        tuple:
        H (cupy.ndarray):
            The multidimensional histogram of sample x. See
            normed and weights for the different possible semantics.
        edges (list of cupy.ndarray):
            A list of D arrays describing the bin
            edges for each dimension.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogramdd`
    """
    if isinstance(sample, cupy.ndarray):
        if sample.ndim == 1:
            sample = sample[:, cupy.newaxis]
        nsamples, ndim = sample.shape
    else:
        sample = cupy.stack(sample, axis=-1)
        nsamples, ndim = sample.shape
    nbin = numpy.empty(ndim, int)
    edges = ndim * [None]
    dedges = ndim * [None]
    if weights is not None:
        weights = cupy.asarray(weights)
    try:
        nbins = len(bins)
        if nbins != ndim:
            raise ValueError('The dimension of bins must be equal to the dimension of the  sample x.')
    except TypeError:
        bins = ndim * [bins]
    if range is None:
        range = (None,) * ndim
    elif len(range) != ndim:
        raise ValueError('range argument must have one entry per dimension')
    for i in _range(ndim):
        if cupy.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError('`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            num = int(bins[i] + 1)
            edges[i] = cupy.linspace(smin, smax, num)
        elif cupy.ndim(bins[i]) == 1:
            if not isinstance(bins[i], cupy.ndarray):
                raise ValueError('array-like bins not supported')
            edges[i] = bins[i]
            if (edges[i][:-1] > edges[i][1:]).any():
                raise ValueError('`bins[{}]` must be monotonically increasing, when an array'.format(i))
        else:
            raise ValueError('`bins[{}]` must be a scalar or 1d array'.format(i))
        nbin[i] = len(edges[i]) + 1
        dedges[i] = cupy.diff(edges[i])
    ncount = tuple((cupy.searchsorted(edges[i], sample[:, i], side='right') for i in _range(ndim)))
    for i in _range(ndim):
        on_edge = sample[:, i] == edges[i][-1]
        ncount[i][on_edge] -= 1
    xy = cupy.ravel_multi_index(ncount, nbin)
    hist = cupy.bincount(xy, weights, minlength=numpy.prod(nbin))
    hist = hist.reshape(nbin)
    hist = hist.astype(float)
    core = ndim * (slice(1, -1),)
    hist = hist[core]
    if density:
        s = hist.sum()
        for i in _range(ndim):
            shape = [1] * ndim
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s
    if any(hist.shape != numpy.asarray(nbin) - 2):
        raise RuntimeError('Internal Shape Error')
    return (hist, edges)