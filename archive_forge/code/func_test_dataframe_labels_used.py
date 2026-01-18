import warnings
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
@pytest.mark.filterwarnings('ignore:X has feature names, but LogisticRegression was fitted without')
def test_dataframe_labels_used(pyplot, fitted_clf):
    """Check that column names are used for pandas."""
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(X, columns=['col_x', 'col_y'])
    _, ax = pyplot.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, df, ax=ax)
    assert ax.get_xlabel() == 'col_x'
    assert ax.get_ylabel() == 'col_y'
    fig, ax = pyplot.subplots()
    disp.plot(ax=ax)
    assert ax.get_xlabel() == 'col_x'
    assert ax.get_ylabel() == 'col_y'
    fig, ax = pyplot.subplots()
    ax.set(xlabel='hello', ylabel='world')
    disp.plot(ax=ax)
    assert ax.get_xlabel() == 'hello'
    assert ax.get_ylabel() == 'world'
    disp.plot(ax=ax, xlabel='overwritten_x', ylabel='overwritten_y')
    assert ax.get_xlabel() == 'overwritten_x'
    assert ax.get_ylabel() == 'overwritten_y'
    _, ax = pyplot.subplots()
    disp = DecisionBoundaryDisplay.from_estimator(fitted_clf, df, ax=ax, xlabel='overwritten_x', ylabel='overwritten_y')
    assert ax.get_xlabel() == 'overwritten_x'
    assert ax.get_ylabel() == 'overwritten_y'