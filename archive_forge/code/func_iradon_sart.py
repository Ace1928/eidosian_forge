import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import golden_ratio
from scipy.fft import fft, ifft, fftfreq, fftshift
from ._warps import warp
from ._radon_transform import sart_projection_update
from .._shared.utils import convert_to_float
from warnings import warn
from functools import partial
def iradon_sart(radon_image, theta=None, image=None, projection_shifts=None, clip=None, relaxation=0.15, dtype=None):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using a single iteration of
    the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Parameters
    ----------
    radon_image : ndarray, shape (M, N)
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array, shape (N,), optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    image : ndarray, shape (M, M), optional
        Image containing an initial reconstruction estimate. Default is an array of zeros.
    projection_shifts : array, shape (N,), optional
        Shift the projections contained in ``radon_image`` (the sinogram) by
        this many pixels before reconstructing the image. The i'th value
        defines the shift of the i'th column of ``radon_image``.
    clip : length-2 sequence of floats, optional
        Force all values in the reconstructed tomogram to lie in the range
        ``[clip[0], clip[1]]``
    relaxation : float, optional
        Relaxation parameter for the update step. A higher value can
        improve the convergence rate, but one runs the risk of instabilities.
        Values close to or higher than 1 are not recommended.
    dtype : dtype, optional
        Output data type, must be floating point. By default, if input
        data type is not float, input is cast to double, otherwise
        dtype is set to input data type.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    Algebraic Reconstruction Techniques are based on formulating the tomography
    reconstruction problem as a set of linear equations. Along each ray,
    the projected value is the sum of all the values of the cross section along
    the ray. A typical feature of SART (and a few other variants of algebraic
    techniques) is that it samples the cross section at equidistant points
    along the ray, using linear interpolation between the pixel values of the
    cross section. The resulting set of linear equations are then solved using
    a slightly modified Kaczmarz method.

    When using SART, a single iteration is usually sufficient to obtain a good
    reconstruction. Further iterations will tend to enhance high-frequency
    information, but will also often increase the noise.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction
           technique (SART): a superior implementation of the ART algorithm",
           Ultrasonic Imaging 6 pp 81--94 (1984)
    .. [3] S Kaczmarz, "Angenäherte auflösung von systemen linearer
           gleichungen", Bulletin International de l’Academie Polonaise des
           Sciences et des Lettres 35 pp 355--357 (1937)
    .. [4] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [5] Kaczmarz' method, Wikipedia,
           https://en.wikipedia.org/wiki/Kaczmarz_method

    """
    if radon_image.ndim != 2:
        raise ValueError('radon_image must be two dimensional')
    if dtype is None:
        if radon_image.dtype.char in 'fd':
            dtype = radon_image.dtype
        else:
            warn('Only floating point data type are valid for SART inverse radon transform. Input data is cast to float. To disable this warning, please cast image_radon to float.')
            dtype = np.dtype(float)
    elif np.dtype(dtype).char not in 'fd':
        raise ValueError('Only floating point data type are valid for inverse radon transform.')
    dtype = np.dtype(dtype)
    radon_image = radon_image.astype(dtype, copy=False)
    reconstructed_shape = (radon_image.shape[0], radon_image.shape[0])
    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False, dtype=dtype)
    elif len(theta) != radon_image.shape[1]:
        raise ValueError(f'Shape of theta ({len(theta)}) does not match the number of projections ({radon_image.shape[1]})')
    else:
        theta = np.asarray(theta, dtype=dtype)
    if image is None:
        image = np.zeros(reconstructed_shape, dtype=dtype)
    elif image.shape != reconstructed_shape:
        raise ValueError(f'Shape of image ({image.shape}) does not match first dimension of radon_image ({reconstructed_shape})')
    elif image.dtype != dtype:
        warn(f'image dtype does not match output dtype: image is cast to {dtype}')
    image = np.asarray(image, dtype=dtype)
    if projection_shifts is None:
        projection_shifts = np.zeros((radon_image.shape[1],), dtype=dtype)
    elif len(projection_shifts) != radon_image.shape[1]:
        raise ValueError(f'Shape of projection_shifts ({len(projection_shifts)}) does not match the number of projections ({radon_image.shape[1]})')
    else:
        projection_shifts = np.asarray(projection_shifts, dtype=dtype)
    if clip is not None:
        if len(clip) != 2:
            raise ValueError('clip must be a length-2 sequence')
        clip = np.asarray(clip, dtype=dtype)
    for angle_index in order_angles_golden_ratio(theta):
        image_update = sart_projection_update(image, theta[angle_index], radon_image[:, angle_index], projection_shifts[angle_index])
        image += relaxation * image_update
        if clip is not None:
            image = np.clip(image, clip[0], clip[1])
    return image