import numpy as np
from numpy.testing import assert_allclose
from scipy import ndimage
from scipy.ndimage import _ctest
from scipy.ndimage import _cytest
from scipy._lib._ccallback import LowLevelCallable
def test_generic_filter1d():

    def filter1d(input_line, output_line, filter_size):
        for i in range(output_line.size):
            output_line[i] = 0
            for j in range(filter_size):
                output_line[i] += input_line[i + j]
        output_line /= filter_size

    def check(j):
        func = FILTER1D_FUNCTIONS[j]
        im = np.tile(np.hstack((np.zeros(10), np.ones(10))), (10, 1))
        filter_size = 3
        res = ndimage.generic_filter1d(im, func(filter_size), filter_size)
        std = ndimage.generic_filter1d(im, filter1d, filter_size, extra_arguments=(filter_size,))
        assert_allclose(res, std, err_msg=f'#{j} failed')
    for j, func in enumerate(FILTER1D_FUNCTIONS):
        check(j)