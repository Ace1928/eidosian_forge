import warnings
import operator
import numpy
from . import _ni_support
from . import _nd_image
from . import _filters
def distance_transform_bf(input, metric='euclidean', sampling=None, return_distances=True, return_indices=False, distances=None, indices=None):
    """
    Distance transform function by a brute force algorithm.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input
    metric : {'euclidean', 'taxicab', 'chessboard'}, optional
        'cityblock' and 'manhattan' are also valid, and map to 'taxicab'.
        The default is 'euclidean'.
    sampling : float, or sequence of float, optional
        This parameter is only used when `metric` is 'euclidean'.
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`, and of type float64 if `metric`
        is 'euclidean', uint32 otherwise.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be `(input.ndim,) + input.shape`.

    Returns
    -------
    distances : ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See distance_transform_edt documentation for an
        example.
        Returned only when `return_indices` is True and `indices` is not
        supplied.

    See Also
    --------
    distance_transform_cdt : Faster distance transform for taxicab and
                             chessboard metrics
    distance_transform_edt : Faster distance transform for euclidean metric

    Notes
    -----
    This function employs a slow brute force algorithm. See also the
    function `distance_transform_cdt` for more efficient taxicab [1]_ and
    chessboard algorithms [2]_.

    References
    ----------
    .. [1] Taxicab distance. Wikipedia, 2023.
           https://en.wikipedia.org/wiki/Taxicab_geometry
    .. [2] Chessboard distance. Wikipedia, 2023.
           https://en.wikipedia.org/wiki/Chebyshev_distance

    Examples
    --------
    Import the necessary modules.

    >>> import numpy as np
    >>> from scipy.ndimage import distance_transform_bf
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1 import ImageGrid

    First, we create a toy binary image.

    >>> def add_circle(center_x, center_y, radius, image, fillvalue=1):
    ...     # fill circular area with 1
    ...     xx, yy = np.mgrid[:image.shape[0], :image.shape[1]]
    ...     circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
    ...     circle_shape = np.sqrt(circle) < radius
    ...     image[circle_shape] = fillvalue
    ...     return image
    >>> image = np.zeros((100, 100), dtype=np.uint8)
    >>> image[35:65, 20:80] = 1
    >>> image = add_circle(28, 65, 10, image)
    >>> image = add_circle(37, 30, 10, image)
    >>> image = add_circle(70, 45, 20, image)
    >>> image = add_circle(45, 80, 10, image)

    Next, we set up the figure.

    >>> fig = plt.figure(figsize=(8, 8))  # set up the figure structure
    >>> grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.4, 0.3),
    ...                  label_mode="1", share_all=True,
    ...                  cbar_location="right", cbar_mode="each",
    ...                  cbar_size="7%", cbar_pad="2%")
    >>> for ax in grid:
    ...     ax.axis('off')  # remove axes from images

    The top left image is the original binary image.

    >>> binary_image = grid[0].imshow(image, cmap='gray')
    >>> cbar_binary_image = grid.cbar_axes[0].colorbar(binary_image)
    >>> cbar_binary_image.set_ticks([0, 1])
    >>> grid[0].set_title("Binary image: foreground in white")

    The distance transform calculates the distance between foreground pixels
    and the image background according to a distance metric. Available metrics
    in `distance_transform_bf` are: ``euclidean`` (default), ``taxicab``
    and ``chessboard``. The top right image contains the distance transform
    based on the ``euclidean`` metric.

    >>> distance_transform_euclidean = distance_transform_bf(image)
    >>> euclidean_transform = grid[1].imshow(distance_transform_euclidean,
    ...                                      cmap='gray')
    >>> cbar_euclidean = grid.cbar_axes[1].colorbar(euclidean_transform)
    >>> colorbar_ticks = [0, 10, 20]
    >>> cbar_euclidean.set_ticks(colorbar_ticks)
    >>> grid[1].set_title("Euclidean distance")

    The lower left image contains the distance transform using the ``taxicab``
    metric.

    >>> distance_transform_taxicab = distance_transform_bf(image,
    ...                                                    metric='taxicab')
    >>> taxicab_transformation = grid[2].imshow(distance_transform_taxicab,
    ...                                         cmap='gray')
    >>> cbar_taxicab = grid.cbar_axes[2].colorbar(taxicab_transformation)
    >>> cbar_taxicab.set_ticks(colorbar_ticks)
    >>> grid[2].set_title("Taxicab distance")

    Finally, the lower right image contains the distance transform using the
    ``chessboard`` metric.

    >>> distance_transform_cb = distance_transform_bf(image,
    ...                                               metric='chessboard')
    >>> chessboard_transformation = grid[3].imshow(distance_transform_cb,
    ...                                            cmap='gray')
    >>> cbar_taxicab = grid.cbar_axes[3].colorbar(chessboard_transformation)
    >>> cbar_taxicab.set_ticks(colorbar_ticks)
    >>> grid[3].set_title("Chessboard distance")
    >>> plt.show()

    """
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(dt_inplace, ft_inplace, return_distances, return_indices)
    tmp1 = numpy.asarray(input) != 0
    struct = generate_binary_structure(tmp1.ndim, tmp1.ndim)
    tmp2 = binary_dilation(tmp1, struct)
    tmp2 = numpy.logical_xor(tmp1, tmp2)
    tmp1 = tmp1.astype(numpy.int8) - tmp2.astype(numpy.int8)
    metric = metric.lower()
    if metric == 'euclidean':
        metric = 1
    elif metric in ['taxicab', 'cityblock', 'manhattan']:
        metric = 2
    elif metric == 'chessboard':
        metric = 3
    else:
        raise RuntimeError('distance metric not supported')
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, tmp1.ndim)
        sampling = numpy.asarray(sampling, dtype=numpy.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()
    if return_indices:
        ft = numpy.zeros(tmp1.shape, dtype=numpy.int32)
    else:
        ft = None
    if return_distances:
        if distances is None:
            if metric == 1:
                dt = numpy.zeros(tmp1.shape, dtype=numpy.float64)
            else:
                dt = numpy.zeros(tmp1.shape, dtype=numpy.uint32)
        else:
            if distances.shape != tmp1.shape:
                raise RuntimeError('distances array has wrong shape')
            if metric == 1:
                if distances.dtype.type != numpy.float64:
                    raise RuntimeError('distances array must be float64')
            elif distances.dtype.type != numpy.uint32:
                raise RuntimeError('distances array must be uint32')
            dt = distances
    else:
        dt = None
    _nd_image.distance_transform_bf(tmp1, metric, sampling, dt, ft)
    if return_indices:
        if isinstance(indices, numpy.ndarray):
            if indices.dtype.type != numpy.int32:
                raise RuntimeError('indices array must be int32')
            if indices.shape != (tmp1.ndim,) + tmp1.shape:
                raise RuntimeError('indices array has wrong shape')
            tmp2 = indices
        else:
            tmp2 = numpy.indices(tmp1.shape, dtype=numpy.int32)
        ft = numpy.ravel(ft)
        for ii in range(tmp2.shape[0]):
            rtmp = numpy.ravel(tmp2[ii, ...])[ft]
            rtmp.shape = tmp1.shape
            tmp2[ii, ...] = rtmp
        ft = tmp2
    result = []
    if return_distances and (not dt_inplace):
        result.append(dt)
    if return_indices and (not ft_inplace):
        result.append(ft)
    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None