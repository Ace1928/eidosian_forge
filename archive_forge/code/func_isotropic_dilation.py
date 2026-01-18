import numpy as np
from scipy import ndimage as ndi
def isotropic_dilation(image, radius, out=None, spacing=None):
    """Return binary morphological dilation of an image.

    This function returns the same result as :func:`skimage.morphology.binary_dilation`
    but performs faster for large circular structuring elements.
    This works by applying a threshold to the exact Euclidean distance map
    of the inverted image [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius by which regions should be dilated.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    dilated : ndarray of bool
        The result of the morphological dilation with values in
        ``[False, True]``.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`
    """
    dist = ndi.distance_transform_edt(np.logical_not(image), sampling=spacing)
    return np.less_equal(dist, radius, out=out)