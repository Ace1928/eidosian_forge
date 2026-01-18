import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
@classmethod
def get_biquartimin_example(cls):
    A = cls.get_A()
    table_required = cls.str2matrix('\n            0.00000    0.21632   -0.54955    1.00000\n            1.00000    0.19519   -0.46174    0.50000\n            2.00000    0.09479   -0.16365    1.00000\n            3.00000   -0.06302   -0.32096    0.50000\n            4.00000   -0.21304   -0.46562    1.00000\n            5.00000   -0.33199   -0.33287    1.00000\n            6.00000   -0.35108   -0.63990    0.12500\n            7.00000   -0.35543   -1.20916    0.12500\n            8.00000   -0.35568   -2.61213    0.12500\n            9.00000   -0.35568   -2.97910    0.06250\n           10.00000   -0.35568   -3.32645    0.06250\n           11.00000   -0.35568   -3.66021    0.06250\n           12.00000   -0.35568   -3.98564    0.06250\n           13.00000   -0.35568   -4.30635    0.06250\n           14.00000   -0.35568   -4.62451    0.06250\n           15.00000   -0.35568   -4.94133    0.06250\n           16.00000   -0.35568   -5.25745    0.06250\n        ')
    L_required = cls.str2matrix('\n           1.01753  -0.13657\n           1.11338  -0.24643\n           1.09200  -0.26890\n           1.00676  -0.16010\n          -0.26534   1.11371\n          -0.26972   0.99553\n          -0.29341   0.93561\n          -0.10806   0.80513\n        ')
    return (A, table_required, L_required)