import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def hough_line(image, theta=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : ndarray of double, shape (K,), optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced in the
        range [-pi/2, pi/2).

    Returns
    -------
    hspace : ndarray of uint64, shape (P, Q)
        Hough transform accumulator.
    angles : ndarray
        Angles at which the transform is computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    The angle accuracy can be improved by decreasing the step size in
    the `theta` array.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> rng = np.random.default_rng()
    >>> img += rng.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)
    """
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')
    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    return _hough_line(image, theta=theta)