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
@pytest.mark.parametrize('constructor', [list, np.array])
def test_export_text_feature_class_names_array_support(constructor):
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    expected_report = dedent('\n    |--- b <= 0.00\n    |   |--- class: -1\n    |--- b >  0.00\n    |   |--- class: 1\n    ').lstrip()
    assert export_text(clf, feature_names=constructor(['a', 'b'])) == expected_report
    expected_report = dedent('\n    |--- feature_1 <= 0.00\n    |   |--- class: cat\n    |--- feature_1 >  0.00\n    |   |--- class: dog\n    ').lstrip()
    assert export_text(clf, class_names=constructor(['cat', 'dog'])) == expected_report