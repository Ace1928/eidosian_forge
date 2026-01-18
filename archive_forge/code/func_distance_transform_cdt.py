import warnings
import operator
import numpy
from . import _ni_support
from . import _nd_image
from . import _filters
def distance_transform_cdt(input, metric='chessboard', return_distances=True, return_indices=False, distances=None, indices=None):
    """
    Distance transform for chamfer type of transforms.

    This function calculates the distance transform of the `input`, by
    replacing each foreground (non-zero) element, with its
    shortest distance to the background (any zero-valued element).

    In addition to the distance transform, the feature transform can
    be calculated. In this case the index of the closest background
    element to each foreground element is returned in a separate array.

    Parameters
    ----------
    input : array_like
        Input. Values of 0 are treated as background.
    metric : {'chessboard', 'taxicab'} or array_like, optional
        The `metric` determines the type of chamfering that is done. If the
        `metric` is equal to 'taxicab' a structure is generated using
        `generate_binary_structure` with a squared distance equal to 1. If
        the `metric` is equal to 'chessboard', a `metric` is generated
        using `generate_binary_structure` with a squared distance equal to
        the dimensionality of the array. These choices correspond to the
        common interpretations of the 'taxicab' and the 'chessboard'
        distance metrics in two dimensions.
        A custom metric may be provided, in the form of a matrix where
        each dimension has a length of three.
        'cityblock' and 'manhattan' are also valid, and map to 'taxicab'.
        The default is 'chessboard'.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : int32 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indicies` must be True.
        Its shape must be `(input.ndim,) + input.shape`.

    Returns
    -------
    distances : int32 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True, and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See distance_transform_edt documentation for an
        example.
        Returned only when `return_indices` is True, and `indices` is not
        supplied.

    See Also
    --------
    distance_transform_edt : Fast distance transform for euclidean metric
    distance_transform_bf : Distance transform for different metrics using
                            a slower brute force algorithm

    Examples
    --------
    Import the necessary modules.

    >>> import numpy as np
    >>> from scipy.ndimage import distance_transform_cdt
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

    >>> fig = plt.figure(figsize=(5, 15))
    >>> grid = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=(0.5, 0.3),
    ...                  label_mode="1", share_all=True,
    ...                  cbar_location="right", cbar_mode="each",
    ...                  cbar_size="7%", cbar_pad="2%")
    >>> for ax in grid:
    ...     ax.axis('off')
    >>> top, middle, bottom = grid
    >>> colorbar_ticks = [0, 10, 20]

    The top image contains the original binary image.

    >>> binary_image = top.imshow(image, cmap='gray')
    >>> cbar_binary_image = top.cax.colorbar(binary_image)
    >>> cbar_binary_image.set_ticks([0, 1])
    >>> top.set_title("Binary image: foreground in white")

    The middle image contains the distance transform using the ``taxicab``
    metric.

    >>> distance_taxicab = distance_transform_cdt(image, metric="taxicab")
    >>> taxicab_transform = middle.imshow(distance_taxicab, cmap='gray')
    >>> cbar_taxicab = middle.cax.colorbar(taxicab_transform)
    >>> cbar_taxicab.set_ticks(colorbar_ticks)
    >>> middle.set_title("Taxicab metric")

    The bottom image contains the distance transform using the ``chessboard``
    metric.

    >>> distance_chessboard = distance_transform_cdt(image,
    ...                                              metric="chessboard")
    >>> chessboard_transform = bottom.imshow(distance_chessboard, cmap='gray')
    >>> cbar_chessboard = bottom.cax.colorbar(chessboard_transform)
    >>> cbar_chessboard.set_ticks(colorbar_ticks)
    >>> bottom.set_title("Chessboard metric")
    >>> plt.tight_layout()
    >>> plt.show()

    """
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(dt_inplace, ft_inplace, return_distances, return_indices)
    input = numpy.asarray(input)
    if isinstance(metric, str):
        if metric in ['taxicab', 'cityblock', 'manhattan']:
            rank = input.ndim
            metric = generate_binary_structure(rank, 1)
        elif metric == 'chessboard':
            rank = input.ndim
            metric = generate_binary_structure(rank, rank)
        else:
            raise ValueError('invalid metric provided')
    else:
        try:
            metric = numpy.asarray(metric)
        except Exception as e:
            raise ValueError('invalid metric provided') from e
        for s in metric.shape:
            if s != 3:
                raise ValueError('metric sizes must be equal to 3')
    if not metric.flags.contiguous:
        metric = metric.copy()
    if dt_inplace:
        if distances.dtype.type != numpy.int32:
            raise ValueError('distances must be of int32 type')
        if distances.shape != input.shape:
            raise ValueError('distances has wrong shape')
        dt = distances
        dt[...] = numpy.where(input, -1, 0).astype(numpy.int32)
    else:
        dt = numpy.where(input, -1, 0).astype(numpy.int32)
    rank = dt.ndim
    if return_indices:
        sz = numpy.prod(dt.shape, axis=0)
        ft = numpy.arange(sz, dtype=numpy.int32)
        ft.shape = dt.shape
    else:
        ft = None
    _nd_image.distance_transform_op(metric, dt, ft)
    dt = dt[tuple([slice(None, None, -1)] * rank)]
    if return_indices:
        ft = ft[tuple([slice(None, None, -1)] * rank)]
    _nd_image.distance_transform_op(metric, dt, ft)
    dt = dt[tuple([slice(None, None, -1)] * rank)]
    if return_indices:
        ft = ft[tuple([slice(None, None, -1)] * rank)]
        ft = numpy.ravel(ft)
        if ft_inplace:
            if indices.dtype.type != numpy.int32:
                raise ValueError('indices array must be int32')
            if indices.shape != (dt.ndim,) + dt.shape:
                raise ValueError('indices array has wrong shape')
            tmp = indices
        else:
            tmp = numpy.indices(dt.shape, dtype=numpy.int32)
        for ii in range(tmp.shape[0]):
            rtmp = numpy.ravel(tmp[ii, ...])[ft]
            rtmp.shape = dt.shape
            tmp[ii, ...] = rtmp
        ft = tmp
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