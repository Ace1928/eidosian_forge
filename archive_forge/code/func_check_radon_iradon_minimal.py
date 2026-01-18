import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def check_radon_iradon_minimal(shape, slices):
    debug = False
    theta = np.arange(180)
    image = np.zeros(shape, dtype=float)
    image[slices] = 1.0
    sinogram = radon(image, theta, circle=False)
    reconstructed = iradon(sinogram, theta, circle=False)
    print('\n\tMaximum deviation:', np.max(np.abs(image - reconstructed)))
    if debug and has_mpl:
        _debug_plot(image, reconstructed, sinogram)
    if image.sum() == 1:
        assert np.unravel_index(np.argmax(reconstructed), image.shape) == np.unravel_index(np.argmax(image), image.shape)