import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_delta_0(self):
    self.generate('test_delta_0', 'test_lowess_delta.csv', out='out_0', kwargs={'frac': 0.1})