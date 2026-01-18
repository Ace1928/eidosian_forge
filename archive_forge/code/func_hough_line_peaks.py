import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10, threshold=None, num_peaks=np.inf):
    """Return peaks in a straight line Hough transform.

    Identifies most prominent lines separated by a certain angle and distance
    in a Hough transform. Non-maximum suppression with different sizes is
    applied separately in the first (distances) and second (angles) dimension
    of the Hough space to identify peaks.

    Parameters
    ----------
    hspace : ndarray, shape (M, N)
        Hough space returned by the `hough_line` function.
    angles : array, shape (N,)
        Angles returned by the `hough_line` function. Assumed to be continuous.
        (`angles[-1] - angles[0] == PI`).
    dists : array, shape (M,)
        Distances returned by the `hough_line` function.
    min_distance : int, optional
        Minimum distance separating lines (maximum filter size for first
        dimension of hough space).
    min_angle : int, optional
        Minimum angle separating lines (maximum filter size for second
        dimension of hough space).
    threshold : float, optional
        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    accum, angles, dists : tuple of array
        Peak values in Hough space, angles and distances.

    Examples
    --------
    >>> from skimage.transform import hough_line, hough_line_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=bool)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough_line(img)
    >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    >>> len(angles)
    2

    """
    from ..feature.peak import _prominent_peaks
    min_angle = min(min_angle, hspace.shape[1])
    h, a, d = _prominent_peaks(hspace, min_xdistance=min_angle, min_ydistance=min_distance, threshold=threshold, num_peaks=num_peaks)
    if a.size > 0:
        return (h, angles[a], dists[d])
    else:
        return (h, np.array([]), np.array([]))