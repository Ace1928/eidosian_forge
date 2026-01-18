import warnings
import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp
def graycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False):
    """Calculate the gray-level co-occurrence matrix.

    A gray level co-occurrence matrix is a histogram of co-occurring
    grayscale values at a given offset over an image.

    .. versionchanged:: 0.19
               `greymatrix` was renamed to `graymatrix` in 0.19.

    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.

    Returns
    -------
    P : 4-D ndarray
        The gray-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that gray-level `j`
        occurs at a distance `d` and at an angle `theta` from
        gray-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`
    .. [2] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for
           image classification", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973.
           :DOI:`10.1109/TSMC.1973.4309314`
    .. [3] M. Nadler and E.P. Smith, Pattern Recognition Engineering,
           Wiley-Interscience, 1993.
    .. [4] Wikipedia, https://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)

    """
    check_nD(image, 2)
    check_nD(distances, 1, 'distances')
    check_nD(angles, 1, 'angles')
    image = np.ascontiguousarray(image)
    image_max = image.max()
    if np.issubdtype(image.dtype, np.floating):
        raise ValueError('Float images are not supported by graycomatrix. Convert the image to an unsigned integer type.')
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError('The levels argument is required for data types other than uint8. The resulting matrix will be at least levels ** 2 in size.')
    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError('Negative-valued images are not supported.')
    if levels is None:
        levels = 256
    if image_max >= levels:
        raise ValueError('The maximum grayscale value in the image should be smaller than the number of levels.')
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)
    P = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32, order='C')
    _glcm_loop(image, distances, angles, levels, P)
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums
    return P