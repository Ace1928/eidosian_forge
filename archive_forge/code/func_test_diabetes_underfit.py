import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@skip_if_32bit
@pytest.mark.parametrize('name, Tree', REG_TREES.items())
@pytest.mark.parametrize('criterion, max_depth, metric, max_loss', [('squared_error', 15, mean_squared_error, 60), ('absolute_error', 20, mean_squared_error, 60), ('friedman_mse', 15, mean_squared_error, 60), ('poisson', 15, mean_poisson_deviance, 30)])
def test_diabetes_underfit(name, Tree, criterion, max_depth, metric, max_loss):
    reg = Tree(criterion=criterion, max_depth=max_depth, max_features=6, random_state=0)
    reg.fit(diabetes.data, diabetes.target)
    loss = metric(diabetes.target, reg.predict(diabetes.data))
    assert 0 < loss < max_loss