import inspect
import os
import numpy as np
import pytest
import sklearn.datasets
@pytest.mark.parametrize('name, dataset_func', _generate_func_supporting_param('as_frame'))
def test_common_check_as_frame(name, dataset_func):
    bunch = dataset_func()
    check_as_frame(bunch, dataset_func)