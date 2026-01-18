import numpy as np
from skimage.restoration import unwrap_phase
import sys
from skimage._shared import testing
from skimage._shared.testing import (
from skimage._shared._warnings import expected_warnings
def check_wrap_around(ndim, axis):
    elements = 100
    ramp = np.linspace(0, 12 * np.pi, elements)
    ramp[-1] = ramp[0]
    image = ramp.reshape(tuple([elements if n == axis else 1 for n in range(ndim)]))
    image_wrapped = np.angle(np.exp(1j * image))
    index_first = tuple([0] * ndim)
    index_last = tuple([-1 if n == axis else 0 for n in range(ndim)])
    with expected_warnings(['Image has a length 1 dimension|\\A\\Z']):
        image_unwrap_no_wrap_around = unwrap_phase(image_wrapped, rng=0)
    print('endpoints without wrap_around:', image_unwrap_no_wrap_around[index_first], image_unwrap_no_wrap_around[index_last])
    assert_(abs(image_unwrap_no_wrap_around[index_first] - image_unwrap_no_wrap_around[index_last]) > np.pi)
    wrap_around = [n == axis for n in range(ndim)]
    with expected_warnings(['Image has a length 1 dimension.|\\A\\Z']):
        image_unwrap_wrap_around = unwrap_phase(image_wrapped, wrap_around, rng=0)
    print('endpoints with wrap_around:', image_unwrap_wrap_around[index_first], image_unwrap_wrap_around[index_last])
    assert_almost_equal(image_unwrap_wrap_around[index_first], image_unwrap_wrap_around[index_last])