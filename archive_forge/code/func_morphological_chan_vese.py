from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
def morphological_chan_vese(image, num_iter, init_level_set='checkerboard', smoothing=1, lambda1=1, lambda2=1, iter_callback=lambda x: None):
    """Morphological Active Contours without Edges (MorphACWE)

    Active contours without edges implemented with morphological operators. It
    can be used to segment objects in images and volumes without well defined
    borders. It is required that the inside of the object looks different on
    average than the outside (i.e., the inner area of the object should be
    darker or lighter than the outer area on average).

    Parameters
    ----------
    image : (M, N) or (L, M, N) array
        Grayscale image or volume to be segmented.
    num_iter : uint
        Number of num_iter to run
    init_level_set : str, (M, N) array, or (L, M, N) array
        Initial level set. If an array is given, it will be binarized and used
        as the initial level set. If a string is given, it defines the method
        to generate a reasonable initial level set with the shape of the
        `image`. Accepted values are 'checkerboard' and 'disk'. See the
        documentation of `checkerboard_level_set` and `disk_level_set`
        respectively for details about how these level sets are created.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    lambda1 : float, optional
        Weight parameter for the outer region. If `lambda1` is larger than
        `lambda2`, the outer region will contain a larger range of values than
        the inner region.
    lambda2 : float, optional
        Weight parameter for the inner region. If `lambda2` is larger than
        `lambda1`, the inner region will contain a larger range of values than
        the outer region.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.

    Returns
    -------
    out : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)

    See Also
    --------
    disk_level_set, checkerboard_level_set

    Notes
    -----
    This is a version of the Chan-Vese algorithm that uses morphological
    operators instead of solving a partial differential equation (PDE) for the
    evolution of the contour. The set of morphological operators used in this
    algorithm are proved to be infinitesimally equivalent to the Chan-Vese PDE
    (see [1]_). However, morphological operators are do not suffer from the
    numerical stability issues typically found in PDEs (it is not necessary to
    find the right time step for the evolution), and are computationally
    faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, :DOI:`10.1109/TPAMI.2013.106`
    """
    init_level_set = _init_level_set(init_level_set, image.shape)
    _check_input(image, init_level_set)
    u = np.int8(init_level_set > 0)
    iter_callback(u)
    for _ in range(num_iter):
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-08)
        c1 = (image * u).sum() / float(u.sum() + 1e-08)
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c1) ** 2 - lambda2 * (image - c0) ** 2)
        u[aux < 0] = 1
        u[aux > 0] = 0
        for _ in range(smoothing):
            u = _curvop(u)
        iter_callback(u)
    return u