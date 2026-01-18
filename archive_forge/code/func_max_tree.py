import numpy as np
from ._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import invert
from . import _max_tree
def max_tree(image, connectivity=1):
    """Build the max tree from an image.

    Component trees represent the hierarchical structure of the connected
    components resulting from sequential thresholding operations applied to an
    image. A connected component at one level is parent of a component at a
    higher level if the latter is included in the first. A max-tree is an
    efficient representation of a component tree. A connected component at
    one level is represented by one reference pixel at this level, which is
    parent to all other pixels at that level and to the reference pixel at the
    level above. The max-tree is the basis for many morphological operators,
    namely connected operators.

    Parameters
    ----------
    image : ndarray
        The input image for which the max-tree is to be calculated.
        This image can be of any type.
    connectivity : unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. In 2D, it is 1 for
        a 4-neighborhood and 2 for a 8-neighborhood. Default value is 1.

    Returns
    -------
    parent : ndarray, int64
        Array of same shape as image. The value of each pixel is the index of
        its parent in the ravelled array.
    tree_traverser : 1D array, int64
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    References
    ----------
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
           :DOI:`10.1109/83.663500`
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
           :DOI:`10.1109/ICIP.2007.4379949`
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
           :DOI:`10.1109/TIP.2006.877518`
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
           :DOI:`10.1109/TIP.2014.2336551`

    Examples
    --------
    We create a small sample image (Figure 1 from [4]) and build the max-tree.

    >>> image = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]])
    >>> P, S = max_tree(image, connectivity=2)
    """
    mask = np.ones(image.shape)
    for k in range(len(image.shape)):
        np.moveaxis(mask, k, 0)[0] = 0
        np.moveaxis(mask, k, 0)[-1] = 0
    neighbors, offset = _validate_connectivity(image.ndim, connectivity, offset=None)
    parent = np.zeros(image.shape, dtype=np.int64)
    flat_neighborhood = _offsets_to_raveled_neighbors(image.shape, neighbors, offset).astype(np.int32)
    tree_traverser = np.argsort(image.ravel(), kind='stable').astype(np.int64)
    _max_tree._max_tree(image.ravel(), mask.ravel().astype(np.uint8), flat_neighborhood, offset.astype(np.int32), np.array(image.shape, dtype=np.int32), parent.ravel(), tree_traverser)
    return (parent, tree_traverser)