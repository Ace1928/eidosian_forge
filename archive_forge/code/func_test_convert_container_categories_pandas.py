import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def test_convert_container_categories_pandas():
    pytest.importorskip('pandas')
    df = _convert_container([['x']], 'dataframe', ['A'], categorical_feature_names=['A'])
    assert df.dtypes.iloc[0] == 'category'