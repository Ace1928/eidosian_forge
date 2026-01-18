import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('transform_output', ['pandas', 'polars'])
def test__get_output_config(transform_output):
    """Check _get_output_config works as expected."""
    global_config = get_config()['transform_output']
    config = _get_output_config('transform')
    assert config['dense'] == global_config
    with config_context(transform_output=transform_output):
        config = _get_output_config('transform')
        assert config['dense'] == transform_output
        est = EstimatorNoSetOutputWithTransform()
        config = _get_output_config('transform', est)
        assert config['dense'] == transform_output
        est = EstimatorWithSetOutput()
        config = _get_output_config('transform', est)
        assert config['dense'] == transform_output
        est.set_output(transform='default')
        config = _get_output_config('transform', est)
        assert config['dense'] == 'default'
    est.set_output(transform=transform_output)
    config = _get_output_config('transform', est)
    assert config['dense'] == transform_output