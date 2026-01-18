import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def check_radon_iradon_circle(interpolation, shape, output_size):
    image = _random_circle(shape)
    radius = min(shape) // 2
    sinogram_rectangle = radon(image, circle=False)
    reconstruction_rectangle = iradon(sinogram_rectangle, output_size=output_size, interpolation=interpolation, circle=False)
    sinogram_circle = radon(image, circle=True)
    reconstruction_circle = iradon(sinogram_circle, output_size=output_size, interpolation=interpolation, circle=True)
    width = reconstruction_circle.shape[0]
    excess = int(np.ceil((reconstruction_rectangle.shape[0] - width) / 2))
    s = np.s_[excess:width + excess, excess:width + excess]
    reconstruction_rectangle = reconstruction_rectangle[s]
    c0, c1 = np.ogrid[0:width, 0:width]
    r = np.sqrt((c0 - width // 2) ** 2 + (c1 - width // 2) ** 2)
    reconstruction_rectangle[r > radius] = 0.0
    print(reconstruction_circle.shape)
    print(reconstruction_rectangle.shape)
    np.allclose(reconstruction_rectangle, reconstruction_circle)