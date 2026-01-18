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
def test_multi_target_sample_weights_api():
    X = [[1, 2, 3], [4, 5, 6]]
    y = [[3.141, 2.718], [2.718, 3.141]]
    w = [0.8, 0.6]
    rgr = MultiOutputRegressor(OrthogonalMatchingPursuit())
    msg = 'does not support sample weights'
    with pytest.raises(ValueError, match=msg):
        rgr.fit(X, y, w)
    rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    rgr.fit(X, y, w)