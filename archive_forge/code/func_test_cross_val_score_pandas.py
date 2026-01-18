import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.filterwarnings('ignore: Using or importing the ABCs from')
def test_cross_val_score_pandas():
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series
        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TargetType, InputFeatureType in types:
        X_df, y_ser = (InputFeatureType(X), TargetType(y2))
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        cross_val_score(clf, X_df, y_ser, cv=3)