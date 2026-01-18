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
def test_multi_output_delegate_predict_proba():
    """Check the behavior for the delegation of predict_proba to the underlying
    estimator"""
    moc = MultiOutputClassifier(LogisticRegression())
    assert hasattr(moc, 'predict_proba')
    moc.fit(X, y)
    assert hasattr(moc, 'predict_proba')
    moc = MultiOutputClassifier(LinearSVC(dual='auto'))
    assert not hasattr(moc, 'predict_proba')
    outer_msg = "'MultiOutputClassifier' has no attribute 'predict_proba'"
    inner_msg = "'LinearSVC' object has no attribute 'predict_proba'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        moc.predict_proba(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg == str(exec_info.value.__cause__)
    moc.fit(X, y)
    assert not hasattr(moc, 'predict_proba')
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        moc.predict_proba(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg == str(exec_info.value.__cause__)