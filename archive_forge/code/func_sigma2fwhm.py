import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def sigma2fwhm(sigma):
    """Convert a sigma in a Gaussian kernel to a FWHM value

    Parameters
    ----------
    sigma : array-like
       sigma value or values

    Returns
    -------
    fwhm : array or float
       fwhm values corresponding to `sigma` values

    Examples
    --------
    >>> fwhm = sigma2fwhm(3)
    >>> fwhms = sigma2fwhm([3, 4, 5])
    >>> fwhm == fwhms[0]
    True
    """
    return np.asarray(sigma) * SIGMA2FWHM