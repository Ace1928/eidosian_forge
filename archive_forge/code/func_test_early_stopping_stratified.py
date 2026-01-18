import re
import sys
import warnings
from io import StringIO
import joblib
import numpy as np
import pytest
from numpy.testing import (
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
def test_early_stopping_stratified():
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 0, 1]
    mlp = MLPClassifier(early_stopping=True)
    with pytest.raises(ValueError, match='The least populated class in y has only 1 member'):
        mlp.fit(X, y)