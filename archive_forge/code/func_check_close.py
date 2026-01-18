import os
from nipype.testing import example_data
import numpy as np
def check_close(val1, val2):
    import numpy.testing as npt
    return npt.assert_almost_equal(val1, val2, decimal=3)