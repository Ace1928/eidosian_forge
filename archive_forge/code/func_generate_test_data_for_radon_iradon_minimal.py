import itertools
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type, convert_to_float
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, rescale
def generate_test_data_for_radon_iradon_minimal(shapes):

    def shape2coordinates(shape):
        c0, c1 = (shape[0] // 2, shape[1] // 2)
        coordinates = itertools.product((c0 - 1, c0, c0 + 1), (c1 - 1, c1, c1 + 1))
        return coordinates

    def shape2shapeandcoordinates(shape):
        return itertools.product([shape], shape2coordinates(shape))
    return itertools.chain.from_iterable([shape2shapeandcoordinates(shape) for shape in shapes])