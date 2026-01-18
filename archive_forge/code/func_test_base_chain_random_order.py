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
def test_base_chain_random_order():
    X, Y = generate_multilabel_dataset_with_correlations()
    for chain in [ClassifierChain(LogisticRegression()), RegressorChain(Ridge())]:
        chain_random = clone(chain).set_params(order='random', random_state=42)
        chain_random.fit(X, Y)
        chain_fixed = clone(chain).set_params(order=chain_random.order_)
        chain_fixed.fit(X, Y)
        assert_array_equal(chain_fixed.order_, chain_random.order_)
        assert list(chain_random.order) != list(range(4))
        assert len(chain_random.order_) == 4
        assert len(set(chain_random.order_)) == 4
        for est1, est2 in zip(chain_random.estimators_, chain_fixed.estimators_):
            assert_array_almost_equal(est1.coef_, est2.coef_)