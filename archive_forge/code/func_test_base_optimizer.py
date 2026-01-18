import numpy as np
from sklearn.neural_network._stochastic_optimizers import (
from sklearn.utils._testing import assert_array_equal
def test_base_optimizer():
    for lr in [10 ** i for i in range(-3, 4)]:
        optimizer = BaseOptimizer(lr)
        assert optimizer.trigger_stopping('', False)