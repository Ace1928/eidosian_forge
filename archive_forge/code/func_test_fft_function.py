import numpy as np
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy
import sys
import pytest
import scipy.fft
def test_fft_function():
    subprocess.check_call([sys.executable, '-c', TEST_BODY])
    from scipy import fft
    assert not callable(fft)
    assert fft.__name__ == 'scipy.fft'
    from scipy import ifft
    assert ifft.__wrapped__ is np.fft.ifft