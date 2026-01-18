import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
def test_load_diabetes_raw():
    """Test to check that we load a scaled version by default but that we can
    get an unscaled version when setting `scaled=False`."""
    diabetes_raw = load_diabetes(scaled=False)
    assert diabetes_raw.data.shape == (442, 10)
    assert diabetes_raw.target.size, 442
    assert len(diabetes_raw.feature_names) == 10
    assert diabetes_raw.DESCR
    diabetes_default = load_diabetes()
    np.testing.assert_allclose(scale(diabetes_raw.data) / 442 ** 0.5, diabetes_default.data, atol=0.0001)