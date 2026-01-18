import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('param', ['data.load', 'stats.information_criterion'])
def test_choice_bad_values(param):
    """Test error messages are correct for rcParams validated with _make_validate_choice."""
    msg = '{}: bad_value is not one of'.format(param.replace('.', '\\.'))
    with pytest.raises(ValueError, match=msg):
        rcParams[param] = 'bad_value'