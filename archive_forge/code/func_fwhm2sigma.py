import numpy as np
import numpy.linalg as npl
from .optpkg import optional_package
from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox
def fwhm2sigma(fwhm):
    """Convert a FWHM value to sigma in a Gaussian kernel.

    Parameters
    ----------
    fwhm : array-like
       FWHM value or values

    Returns
    -------
    sigma : array or float
       sigma values corresponding to `fwhm` values

    Examples
    --------
    >>> sigma = fwhm2sigma(6)
    >>> sigmae = fwhm2sigma([6, 7, 8])
    >>> sigma == sigmae[0]
    True
    """
    return np.asarray(fwhm) / SIGMA2FWHM