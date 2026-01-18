import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def get_random_integer_x_three_classes_y(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    X2 = rng.randint(5, size=(6, 100))
    y2 = np.array([1, 1, 2, 2, 3, 3])
    return (X2, y2)