import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_clone_keeps_output_config():
    """Check that clone keeps the set_output config."""
    ss = StandardScaler().set_output(transform='pandas')
    config = _get_output_config('transform', ss)
    ss_clone = clone(ss)
    config_clone = _get_output_config('transform', ss_clone)
    assert config == config_clone