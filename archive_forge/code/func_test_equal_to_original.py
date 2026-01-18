import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import fft, pi
def test_equal_to_original(self):
    """ Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14) """
    from numpy.core import asarray, concatenate, arange, take

    def original_fftshift(x, axes=None):
        """ How fftshift was implemented in v1.14"""
        tmp = asarray(x)
        ndim = tmp.ndim
        if axes is None:
            axes = list(range(ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        y = tmp
        for k in axes:
            n = tmp.shape[k]
            p2 = (n + 1) // 2
            mylist = concatenate((arange(p2, n), arange(p2)))
            y = take(y, mylist, k)
        return y

    def original_ifftshift(x, axes=None):
        """ How ifftshift was implemented in v1.14 """
        tmp = asarray(x)
        ndim = tmp.ndim
        if axes is None:
            axes = list(range(ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        y = tmp
        for k in axes:
            n = tmp.shape[k]
            p2 = n - (n + 1) // 2
            mylist = concatenate((arange(p2, n), arange(p2)))
            y = take(y, mylist, k)
        return y
    for i in range(16):
        for j in range(16):
            for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                inp = np.random.rand(i, j)
                assert_array_almost_equal(fft.fftshift(inp, axes_keyword), original_fftshift(inp, axes_keyword))
                assert_array_almost_equal(fft.ifftshift(inp, axes_keyword), original_ifftshift(inp, axes_keyword))