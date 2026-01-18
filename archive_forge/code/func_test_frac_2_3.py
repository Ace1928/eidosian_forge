import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_frac_2_3(self):
    self.generate('test_frac_2_3', 'test_lowess_frac.csv', out='out_2_3', kwargs={'frac': 2.0 / 3})