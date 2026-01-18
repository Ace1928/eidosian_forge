from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
def morphological_geodesic_active_contour(gimage, num_iter, init_level_set='disk', smoothing=1, threshold='auto', balloon=0, iter_callback=lambda x: None):
    """Morphological Geodesic Active Contours (MorphGAC).

    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken
    borders.

    Parameters
    ----------
    gimage : (M, N) or (L, M, N) array
        Preprocessed image or volume to be segmented. This is very rarely the
        original image. Instead, this is usually a preprocessed version of the
        original image that enhances and highlights the borders (or other
        structures) of the object to segment.
        :func:`morphological_geodesic_active_contour` will try to stop the contour
        evolution in areas where `gimage` is small. See
        :func:`inverse_gaussian_gradient` as an example function to
        perform this preprocessing. Note that the quality of
        :func:`morphological_geodesic_active_contour` might greatly depend on this
        preprocessing.
    num_iter : uint
        Number of num_iter to run.
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
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in these
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
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
    inverse_gaussian_gradient, disk_level_set, checkerboard_level_set

    Notes
    -----
    This is a version of the Geodesic Active Contours (GAC) algorithm that uses
    morphological operators instead of solving partial differential equations
    (PDEs) for the evolution of the contour. The set of morphological operators
    used in this algorithm are proved to be infinitesimally equivalent to the
    GAC PDEs (see [1]_). However, morphological operators are do not suffer
    from the numerical stability issues typically found in PDEs (e.g., it is
    not necessary to find the right time step for the evolution), and are
    computationally faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, :DOI:`10.1109/TPAMI.2013.106`
    """
    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)
    _check_input(image, init_level_set)
    if threshold == 'auto':
        threshold = np.percentile(image, 40)
    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)
    u = np.int8(init_level_set > 0)
    iter_callback(u)
    for _ in range(num_iter):
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0
        for _ in range(smoothing):
            u = _curvop(u)
        iter_callback(u)
    return u