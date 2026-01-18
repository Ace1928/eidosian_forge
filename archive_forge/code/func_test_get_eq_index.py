from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_get_eq_index(self):
    assert type(self.res.names) is list
    for i, name in enumerate(self.names):
        idx = self.res.get_eq_index(i)
        idx2 = self.res.get_eq_index(name)
        assert_equal(idx, i)
        assert_equal(idx, idx2)
    with pytest.raises(Exception):
        self.res.get_eq_index('foo')