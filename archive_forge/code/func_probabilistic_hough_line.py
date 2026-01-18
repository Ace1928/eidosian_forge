import numpy as np
from scipy.spatial import cKDTree
from ._hough_transform import _hough_circle, _hough_ellipse, _hough_line
from ._hough_transform import _probabilistic_hough_line as _prob_hough_line
def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10, theta=None, rng=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : ndarray of dtype, shape (K,), optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced in the
        range [-pi/2, pi/2).
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y1)),
      indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')
    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    return _prob_hough_line(image, threshold=threshold, line_length=line_length, line_gap=line_gap, theta=theta, rng=rng)