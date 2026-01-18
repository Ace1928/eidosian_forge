from io import StringIO
from re import finditer, search
from textwrap import dedent
import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
def test_graphviz_errors():
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    out = StringIO()
    with pytest.raises(NotFittedError):
        export_graphviz(clf, out)
    clf.fit(X, y)
    message = 'Length of feature_names, 1 does not match number of features, 2'
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=['a'])
    message = 'Length of feature_names, 3 does not match number of features, 2'
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=['a', 'b', 'c'])
    message = 'is not an estimator instance'
    with pytest.raises(TypeError, match=message):
        export_graphviz(clf.fit(X, y).tree_)
    out = StringIO()
    with pytest.raises(IndexError):
        export_graphviz(clf, out, class_names=[])