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
def test_multi_output_exceptions():
    moc = MultiOutputClassifier(LinearSVC(dual='auto', random_state=0))
    with pytest.raises(NotFittedError):
        moc.score(X, y)
    y_new = np.column_stack((y1, y2))
    moc.fit(X, y)
    with pytest.raises(ValueError):
        moc.score(X, y_new)
    msg = 'Unknown label type'
    with pytest.raises(ValueError, match=msg):
        moc.fit(X, X[:, 1])