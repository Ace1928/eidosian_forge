import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
@fails_if_pypy
@pytest.mark.parametrize('parser', ['liac-arff', 'pandas'])
@pytest.mark.parametrize('target_column', ['petalwidth', ['petalwidth', 'petallength']])
def test_fetch_openml_forcing_targets(monkeypatch, parser, target_column):
    """Check that we can force the target to not be the default target."""
    pd = pytest.importorskip('pandas')
    data_id = 61
    _monkey_patch_webbased_functions(monkeypatch, data_id, True)
    bunch_forcing_target = fetch_openml(data_id=data_id, as_frame=True, cache=False, target_column=target_column, parser=parser)
    bunch_default = fetch_openml(data_id=data_id, as_frame=True, cache=False, parser=parser)
    pd.testing.assert_frame_equal(bunch_forcing_target.frame, bunch_default.frame)
    if isinstance(target_column, list):
        pd.testing.assert_index_equal(bunch_forcing_target.target.columns, pd.Index(target_column))
        assert bunch_forcing_target.data.shape == (150, 3)
    else:
        assert bunch_forcing_target.target.name == target_column
        assert bunch_forcing_target.data.shape == (150, 4)