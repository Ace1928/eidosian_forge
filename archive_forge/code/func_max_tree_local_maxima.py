import numpy as np
from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert
from . import _max_tree
def max_tree_local_maxima(image, connectivity=1, parent=None, tree_traverser=None):
    """Determine all local maxima of the image.

    The local maxima are defined as connected sets of pixels with equal
    gray level strictly greater than the gray levels of all pixels in direct
    neighborhood of the set. The function labels the local maxima.

    Technically, the implementation is based on the max-tree representation
    of an image. The function is very efficient if the max-tree representation
    has already been computed. Otherwise, it is preferable to use
    the function local_maxima.

    Parameters
    ----------
    image : ndarray
        The input image for which the maxima are to be calculated.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        The value of each pixel is the index of its parent in the ravelled
        array.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    local_max : ndarray, uint64
        Labeled local maxima of the image.

    See Also
    --------
    skimage.morphology.local_maxima
    skimage.morphology.max_tree

    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
           :DOI:`10.1007/978-3-662-05088-0`
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(int)

    We can calculate all local maxima:

    >>> maxima = max_tree_local_maxima(f)

    The resulting image contains the labeled local maxima.
    """
    output = np.ones(image.shape, dtype=np.uint64)
    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image, connectivity)
    _max_tree._max_tree_local_maxima(image.ravel(), output.ravel(), parent.ravel(), tree_traverser)
    return output