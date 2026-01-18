import warnings
import numpy as np
from scipy.spatial import cKDTree
def hausdorff_distance(image0, image1, method='standard'):
    """Calculate the Hausdorff distance between nonzero elements of given images.

    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.

    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidean distance.

    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155

    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0

    """
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf
    fwd, bwd = (cKDTree(a_points).query(b_points, k=1)[0], cKDTree(b_points).query(a_points, k=1)[0])
    if method == 'standard':
        return max(max(fwd), max(bwd))
    elif method == 'modified':
        return max(np.mean(fwd), np.mean(bwd))