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
def test_multiclass_multioutput_estimator_predict_proba():
    seed = 542
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(5, 5))
    y1 = np.array(['b', 'a', 'a', 'b', 'a']).reshape(5, 1)
    y2 = np.array(['d', 'e', 'f', 'e', 'd']).reshape(5, 1)
    Y = np.concatenate([y1, y2], axis=1)
    clf = MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=seed))
    clf.fit(X, Y)
    y_result = clf.predict_proba(X)
    y_actual = [np.array([[0.23481764, 0.76518236], [0.67196072, 0.32803928], [0.54681448, 0.45318552], [0.34883923, 0.65116077], [0.73687069, 0.26312931]]), np.array([[0.5171785, 0.23878628, 0.24403522], [0.22141451, 0.64102704, 0.13755846], [0.16751315, 0.18256843, 0.64991843], [0.27357372, 0.55201592, 0.17441036], [0.65745193, 0.26062899, 0.08191907]])]
    for i in range(len(y_actual)):
        assert_almost_equal(y_result[i], y_actual[i])