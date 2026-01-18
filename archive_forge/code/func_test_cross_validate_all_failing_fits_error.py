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
@pytest.mark.parametrize('error_score', [np.nan, 0])
def test_cross_validate_all_failing_fits_error(error_score):
    failing_clf = FailingClassifier(FailingClassifier.FAILING_PARAMETER)
    X = np.arange(1, 10)
    y = np.ones(9)
    cross_validate_args = [failing_clf, X, y]
    cross_validate_kwargs = {'cv': 7, 'error_score': error_score}
    individual_fit_error_message = 'ValueError: Failing classifier failed as required'
    error_message = re.compile(f'All the 7 fits failed.+your model is misconfigured.+{individual_fit_error_message}', flags=re.DOTALL)
    with pytest.raises(ValueError, match=error_message):
        cross_validate(*cross_validate_args, **cross_validate_kwargs)