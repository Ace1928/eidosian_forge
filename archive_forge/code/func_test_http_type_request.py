import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_http_type_request(monkeypatch):

    def _urlretrive(url, _):
        raise ValueError(f'URL Retrieved: {url}')
    monkeypatch.setattr(datasets, 'urlretrieve', _urlretrive)
    with pytest.raises(ValueError) as error:
        datasets.load_arviz_data('radon')
        assert 'https://' in str(error)
    with pytest.raises(ValueError) as error:
        rcParams['data.http_protocol'] = 'http'
        datasets.load_arviz_data('radon')
        assert 'http://' in str(error)