import numpy as np
from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert
from . import _max_tree
def area_opening(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None):
    """Perform an area opening of the image.

    Area opening removes all bright structures of an image with
    a surface smaller than area_threshold.
    The output image is thus the largest image smaller than the input
    for which all local maxima have at least a surface of
    area_threshold pixels.

    Area openings are similar to morphological openings, but
    they do not use a fixed footprint, but rather a deformable
    one, with surface = area_threshold. Consequently, the area_opening
    with area_threshold=1 is the identity.

    In the binary case, area openings are equivalent to
    remove_small_objects; this operator is thus extended to gray-level images.

    Technically, this operator is based on the max-tree representation of
    the image.

    Parameters
    ----------
    image : ndarray
        The input image for which the area_opening is to be calculated.
        This image can be of any type.
    area_threshold : unsigned int
        The size parameter (number of pixels). The default value is arbitrarily
        chosen to be 64.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.
    parent : ndarray, int64, optional
        Parent image representing the max tree of the image. The
        value of each pixel is the index of its parent in the ravelled array.
    tree_traverser : 1D array, int64, optional
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    Returns
    -------
    output : ndarray
        Output image of the same shape and type as the input image.

    See Also
    --------
    skimage.morphology.area_closing
    skimage.morphology.diameter_opening
    skimage.morphology.diameter_closing
    skimage.morphology.max_tree
    skimage.morphology.remove_small_objects
    skimage.morphology.remove_small_holes

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
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(int)

    We can calculate the area opening:

    >>> open = area_opening(f, 8, connectivity=1)

    The peaks with a surface smaller than 8 are removed.
    """
    output = image.copy()
    if parent is None or tree_traverser is None:
        parent, tree_traverser = max_tree(image, connectivity)
    area = _max_tree._compute_area(image.ravel(), parent.ravel(), tree_traverser)
    _max_tree._direct_filter(image.ravel(), output.ravel(), parent.ravel(), tree_traverser, area, area_threshold)
    return output