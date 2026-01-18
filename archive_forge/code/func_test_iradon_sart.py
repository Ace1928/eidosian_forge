import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
@run_in_parallel()
def test_iradon_sart():
    debug = False
    image = rescale(PHANTOM, 0.8, mode='reflect', channel_axis=None, anti_aliasing=False)
    theta_ordered = np.linspace(0.0, 180.0, image.shape[0], endpoint=False)
    theta_missing_wedge = np.linspace(0.0, 150.0, image.shape[0], endpoint=True)
    for theta, error_factor in ((theta_ordered, 1.0), (theta_missing_wedge, 2.0)):
        sinogram = radon(image, theta, circle=True)
        reconstructed = iradon_sart(sinogram, theta)
        if debug and has_mpl:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(221)
            plt.imshow(image, interpolation='nearest')
            plt.subplot(222)
            plt.imshow(sinogram, interpolation='nearest')
            plt.subplot(223)
            plt.imshow(reconstructed, interpolation='nearest')
            plt.subplot(224)
            plt.imshow(reconstructed - image, interpolation='nearest')
            plt.show()
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration) =', delta)
        assert delta < 0.02 * error_factor
        reconstructed = iradon_sart(sinogram, theta, reconstructed)
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (2 iterations) =', delta)
        assert delta < 0.014 * error_factor
        reconstructed = iradon_sart(sinogram, theta, clip=(0, 1))
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration, clip) =', delta)
        assert delta < 0.018 * error_factor
        np.random.seed(1239867)
        shifts = np.random.uniform(-3, 3, sinogram.shape[1])
        x = np.arange(sinogram.shape[0])
        sinogram_shifted = np.vstack([np.interp(x + shifts[i], x, sinogram[:, i]) for i in range(sinogram.shape[1])]).T
        reconstructed = iradon_sart(sinogram_shifted, theta, projection_shifts=shifts)
        if debug and has_mpl:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(221)
            plt.imshow(image, interpolation='nearest')
            plt.subplot(222)
            plt.imshow(sinogram_shifted, interpolation='nearest')
            plt.subplot(223)
            plt.imshow(reconstructed, interpolation='nearest')
            plt.subplot(224)
            plt.imshow(reconstructed - image, interpolation='nearest')
            plt.show()
        delta = np.mean(np.abs(reconstructed - image))
        print('delta (1 iteration, shifted sinogram) =', delta)
        assert delta < 0.022 * error_factor