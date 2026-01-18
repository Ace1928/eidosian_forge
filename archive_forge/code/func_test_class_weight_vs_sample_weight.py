import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('reg', (RidgeClassifier, RidgeClassifierCV))
def test_class_weight_vs_sample_weight(reg):
    """Check class_weights resemble sample_weights behavior."""
    reg1 = reg()
    reg1.fit(iris.data, iris.target)
    reg2 = reg(class_weight='balanced')
    reg2.fit(iris.data, iris.target)
    assert_almost_equal(reg1.coef_, reg2.coef_)
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight)
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target)
    assert_almost_equal(reg1.coef_, reg2.coef_)
    reg1 = reg()
    reg1.fit(iris.data, iris.target, sample_weight ** 2)
    reg2 = reg(class_weight=class_weight)
    reg2.fit(iris.data, iris.target, sample_weight)
    assert_almost_equal(reg1.coef_, reg2.coef_)