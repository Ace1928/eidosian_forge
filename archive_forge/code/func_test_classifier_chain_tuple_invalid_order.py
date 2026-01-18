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
def test_classifier_chain_tuple_invalid_order():
    X = [[1, 2, 3], [4, 5, 6], [1.5, 2.5, 3.5]]
    y = [[3, 2], [2, 3], [3, 2]]
    order = tuple([1, 2])
    chain = ClassifierChain(RandomForestClassifier(), order=order)
    with pytest.raises(ValueError, match='invalid order'):
        chain.fit(X, y)