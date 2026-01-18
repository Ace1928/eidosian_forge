import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_frac_1_5(self):
    self.generate('test_frac_1_5', 'test_lowess_frac.csv', out='out_1_5', kwargs={'frac': 1.0 / 5})