import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def stereo_motorcycle():
    """Rectified stereo image pair with ground-truth disparities.

    The two images are rectified such that every pixel in the left image has
    its corresponding pixel on the same scanline in the right image. That means
    that both images are warped such that they have the same orientation but a
    horizontal spatial offset (baseline). The ground-truth pixel offset in
    column direction is specified by the included disparity map.

    The two images are part of the Middlebury 2014 stereo benchmark. The
    dataset was created by Nera Nesic, Porter Westling, Xi Wang, York Kitajima,
    Greg Krathwohl, and Daniel Scharstein at Middlebury College. A detailed
    description of the acquisition process can be found in [1]_.

    The images included here are down-sampled versions of the default exposure
    images in the benchmark. The images are down-sampled by a factor of 4 using
    the function `skimage.transform.downscale_local_mean`. The calibration data
    in the following and the included ground-truth disparity map are valid for
    the down-sampled images::

        Focal length:           994.978px
        Principal point x:      311.193px
        Principal point y:      254.877px
        Principal point dx:      31.086px
        Baseline:               193.001mm

    Returns
    -------
    img_left : (500, 741, 3) uint8 ndarray
        Left stereo image.
    img_right : (500, 741, 3) uint8 ndarray
        Right stereo image.
    disp : (500, 741, 3) float ndarray
        Ground-truth disparity map, where each value describes the offset in
        column direction between corresponding pixels in the left and the right
        stereo images. E.g. the corresponding pixel of
        ``img_left[10, 10 + disp[10, 10]]`` is ``img_right[10, 10]``.
        NaNs denote pixels in the left image that do not have ground-truth.

    Notes
    -----
    The original resolution images, images with different exposure and
    lighting, and ground-truth depth maps can be found at the Middlebury
    website [2]_.

    References
    ----------
    .. [1] D. Scharstein, H. Hirschmueller, Y. Kitajima, G. Krathwohl, N.
           Nesic, X. Wang, and P. Westling. High-resolution stereo datasets
           with subpixel-accurate ground truth. In German Conference on Pattern
           Recognition (GCPR 2014), Muenster, Germany, September 2014.
    .. [2] http://vision.middlebury.edu/stereo/data/scenes2014/

    """
    filename = _fetch('data/motorcycle_disp.npz')
    disp = np.load(filename)['arr_0']
    return (_load('data/motorcycle_left.png'), _load('data/motorcycle_right.png'), disp)