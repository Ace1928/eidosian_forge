import unittest
import numpy as np
from statsmodels.multivariate.factor_rotation._wrappers import rotate_factors
from statsmodels.multivariate.factor_rotation._gpa_rotation import (
from statsmodels.multivariate.factor_rotation._analytic_rotation import (
@classmethod
def get_quartimin_example(cls):
    A = cls.get_A()
    table_required = cls.str2matrix('\n          0.00000    0.42806   -0.46393    1.00000\n        1.00000    0.41311   -0.57313    0.25000\n        2.00000    0.38238   -0.36652    0.50000\n        3.00000    0.31850   -0.21011    0.50000\n        4.00000    0.20937   -0.13838    0.50000\n        5.00000    0.12379   -0.35583    0.25000\n        6.00000    0.04289   -0.53244    0.50000\n        7.00000    0.01098   -0.86649    0.50000\n        8.00000    0.00566   -1.65798    0.50000\n        9.00000    0.00558   -2.13212    0.25000\n       10.00000    0.00557   -2.49020    0.25000\n       11.00000    0.00557   -2.84585    0.25000\n       12.00000    0.00557   -3.20320    0.25000\n       13.00000    0.00557   -3.56143    0.25000\n       14.00000    0.00557   -3.92005    0.25000\n       15.00000    0.00557   -4.27885    0.25000\n       16.00000    0.00557   -4.63772    0.25000\n       17.00000    0.00557   -4.99663    0.25000\n       18.00000    0.00557   -5.35555    0.25000\n        ')
    L_required = cls.str2matrix('\n       0.891822   0.056015\n       0.953680  -0.023246\n       0.929150  -0.046503\n       0.876683   0.033658\n       0.013701   0.925000\n      -0.017265   0.821253\n      -0.052445   0.764953\n       0.085890   0.683115\n        ')
    return (A, table_required, L_required)