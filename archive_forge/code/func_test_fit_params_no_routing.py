import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('Cls, method', [(ClassifierChain, 'fit'), (MultiOutputClassifier, 'partial_fit')])
def test_fit_params_no_routing(Cls, method):
    """Check that we raise an error when passing metadata not requested by the
    underlying classifier.
    """
    X, y = make_classification(n_samples=50)
    clf = Cls(PassiveAggressiveClassifier())
    with pytest.raises(ValueError, match='is only supported if'):
        getattr(clf, method)(X, y, test=1)