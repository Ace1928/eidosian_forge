from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_allclose
import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float
def test_localvar():
    seed = 23703
    data = np.zeros((128, 128)) + 0.5
    local_vars = np.zeros((128, 128)) + 0.001
    local_vars[:64, 64:] = 0.1
    local_vars[64:, :64] = 0.25
    local_vars[64:, 64:] = 0.45
    data_gaussian = random_noise(data, mode='localvar', rng=seed, local_vars=local_vars, clip=False)
    assert 0.0 < data_gaussian[:64, :64].var() < 0.002
    assert 0.095 < data_gaussian[:64, 64:].var() < 0.105
    assert 0.245 < data_gaussian[64:, :64].var() < 0.255
    assert 0.445 < data_gaussian[64:, 64:].var() < 0.455
    bad_local_vars = np.zeros_like(data)
    with testing.raises(ValueError):
        random_noise(data, mode='localvar', rng=seed, local_vars=bad_local_vars)
    bad_local_vars += 0.1
    bad_local_vars[0, 0] = -1
    with testing.raises(ValueError):
        random_noise(data, mode='localvar', rng=seed, local_vars=bad_local_vars)