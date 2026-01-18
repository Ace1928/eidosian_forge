import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def check_radon_iradon(interpolation_type, filter_type):
    debug = False
    image = PHANTOM
    reconstructed = iradon(radon(image, circle=False), filter_name=filter_type, interpolation=interpolation_type, circle=False)
    delta = np.mean(np.abs(image - reconstructed))
    print('\n\tmean error:', delta)
    if debug and has_mpl:
        _debug_plot(image, reconstructed)
    if filter_type in ('ramp', 'shepp-logan'):
        if interpolation_type == 'nearest':
            allowed_delta = 0.03
        else:
            allowed_delta = 0.025
    else:
        allowed_delta = 0.05
    assert delta < allowed_delta