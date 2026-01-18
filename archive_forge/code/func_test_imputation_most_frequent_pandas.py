import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('dtype', [object, 'category'])
def test_imputation_most_frequent_pandas(dtype):
    pd = pytest.importorskip('pandas')
    f = io.StringIO('Cat1,Cat2,Cat3,Cat4\n,i,x,\na,,y,\na,j,,\nb,j,x,')
    df = pd.read_csv(f, dtype=dtype)
    X_true = np.array([['a', 'i', 'x'], ['a', 'j', 'y'], ['a', 'j', 'x'], ['b', 'j', 'x']], dtype=object)
    imputer = SimpleImputer(strategy='most_frequent')
    X_trans = imputer.fit_transform(df)
    assert_array_equal(X_trans, X_true)