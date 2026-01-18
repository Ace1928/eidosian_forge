import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.optimize import minimize, root
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('glm', [TweedieRegressor(power=3), PoissonRegressor(), GammaRegressor(), TweedieRegressor(power=1.5)])
def test_glm_wrong_y_range(glm):
    y = np.array([-1, 2])
    X = np.array([[1], [1]])
    msg = 'Some value\\(s\\) of y are out of the valid range of the loss'
    with pytest.raises(ValueError, match=msg):
        glm.fit(X, y)