import numpy as np
from .._shared.utils import _supported_float_type
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan
def rolling_ball(image, *, radius=100, kernel=None, nansafe=False, num_threads=None):
    """Estimate background intensity by rolling/translating a kernel.

    This rolling ball algorithm estimates background intensity for a
    ndimage in case of uneven exposure. It is a generalization of the
    frequently used rolling ball algorithm [1]_.

    Parameters
    ----------
    image : ndarray
        The image to be filtered.
    radius : int, optional
        Radius of a ball-shaped kernel to be rolled/translated in the image.
        Used if ``kernel = None``.
    kernel : ndarray, optional
        The kernel to be rolled/translated in the image. It must have the
        same number of dimensions as ``image``. Kernel is filled with the
        intensity of the kernel at that position.
    nansafe: bool, optional
        If ``False`` (default) assumes that none of the values in ``image``
        are ``np.nan``, and uses a faster implementation.
    num_threads: int, optional
        The maximum number of threads to use. If ``None`` use the OpenMP
        default value; typically equal to the maximum number of virtual cores.
        Note: This is an upper limit to the number of threads. The exact number
        is determined by the system's OpenMP library.

    Returns
    -------
    background : ndarray
        The estimated background of the image.

    Notes
    -----
    For the pixel that has its background intensity estimated (without loss
    of generality at ``center``) the rolling ball method centers ``kernel``
    under it and raises the kernel until the surface touches the image umbra
    at some ``pos=(y,x)``. The background intensity is then estimated
    using the image intensity at that position (``image[pos]``) plus the
    difference of ``kernel[center] - kernel[pos]``.

    This algorithm assumes that dark pixels correspond to the background. If
    you have a bright background, invert the image before passing it to the
    function, e.g., using `utils.invert`. See the gallery example for details.

    This algorithm is sensitive to noise (in particular salt-and-pepper
    noise). If this is a problem in your image, you can apply mild
    gaussian smoothing before passing the image to this function.

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.restoration import rolling_ball
    >>> image = data.coins()
    >>> background = rolling_ball(data.coins())
    >>> filtered_image = image - background


    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.restoration import rolling_ball, ellipsoid_kernel
    >>> image = data.coins()
    >>> kernel = ellipsoid_kernel((101, 101), 75)
    >>> background = rolling_ball(data.coins(), kernel=kernel)
    >>> filtered_image = image - background
    """
    image = np.asarray(image)
    float_type = _supported_float_type(image.dtype)
    img = image.astype(float_type, copy=False)
    if num_threads is None:
        num_threads = 0
    if kernel is None:
        kernel = ball_kernel(radius, image.ndim)
    kernel = kernel.astype(float_type)
    kernel_shape = np.asarray(kernel.shape)
    kernel_center = kernel_shape // 2
    center_intensity = kernel[tuple(kernel_center)]
    intensity_difference = center_intensity - kernel
    intensity_difference[kernel == np.inf] = np.inf
    intensity_difference = intensity_difference.astype(img.dtype)
    intensity_difference = intensity_difference.reshape(-1)
    img = np.pad(img, kernel_center[:, np.newaxis], constant_values=np.inf, mode='constant')
    func = apply_kernel_nan if nansafe else apply_kernel
    background = func(img.reshape(-1), intensity_difference, np.zeros_like(image, dtype=img.dtype).reshape(-1), np.array(image.shape, dtype=np.intp), np.array(img.shape, dtype=np.intp), kernel_shape.astype(np.intp), num_threads)
    background = background.astype(image.dtype, copy=False)
    return background