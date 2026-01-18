from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('var_names', [None, 'mu', ['mu', 'tau']])
@pytest.mark.parametrize('groups', [None, 'posterior_groups', 'prior_groups', ['posterior', 'sample_stats']])
@pytest.mark.parametrize('dimensions', [None, 'draw', ['chain', 'draw']])
@pytest.mark.parametrize('group_info', [True, False])
@pytest.mark.parametrize('var_name_format', [None, 'brackets', 'underscore', 'cds', ((',', '[', ']'), ('_', ''))])
@pytest.mark.parametrize('index_origin', [None, 0, 1])
def test_flatten_inference_data_to_dict(inference_data, var_names, groups, dimensions, group_info, var_name_format, index_origin):
    """Test flattening (stacking) inference data (subgroups) for dictionary."""
    res_dict = flatten_inference_data_to_dict(data=inference_data, var_names=var_names, groups=groups, dimensions=dimensions, group_info=group_info, var_name_format=var_name_format, index_origin=index_origin)
    assert res_dict
    assert 'draw' in res_dict
    assert any(('mu' in item for item in res_dict))
    if group_info:
        if groups != 'prior_groups':
            assert any(('posterior' in item for item in res_dict))
            if var_names is None:
                assert any(('sample_stats' in item for item in res_dict))
        else:
            assert any(('prior' in item for item in res_dict))
    elif groups == 'prior_groups':
        assert all(('prior' not in item for item in res_dict))
    else:
        assert all(('posterior' not in item for item in res_dict))
        if var_names is None:
            assert all(('sample_stats' not in item for item in res_dict))