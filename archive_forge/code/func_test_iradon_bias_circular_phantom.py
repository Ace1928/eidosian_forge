import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def test_iradon_bias_circular_phantom():
    """
    test that a uniform circular phantom has a small reconstruction bias
    """
    pixels = 128
    xy = np.arange(-pixels / 2, pixels / 2) + 0.5
    x, y = np.meshgrid(xy, xy)
    image = x ** 2 + y ** 2 <= (pixels / 4) ** 2
    theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    reconstruction_fbp = iradon(sinogram, theta=theta)
    error = reconstruction_fbp - image
    tol = 5e-05
    roi_err = np.abs(np.mean(error))
    assert roi_err < tol