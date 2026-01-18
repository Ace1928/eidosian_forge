import importlib
import os
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit
import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical
from ... import (
from ...data.base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
@pytest.fixture(autouse=True)
def no_remote_data(monkeypatch, tmpdir):
    """Delete all remote data and replace it with a local dataset."""
    keys = list(REMOTE_DATASETS)
    for key in keys:
        monkeypatch.delitem(REMOTE_DATASETS, key)
    centered = LOCAL_DATASETS['centered_eight']
    filename = os.path.join(str(tmpdir), os.path.basename(centered.filename))
    url = urlunsplit(('file', '', centered.filename, '', ''))
    monkeypatch.setitem(REMOTE_DATASETS, 'test_remote', RemoteFileMetadata(name='test_remote', filename=filename, url=url, checksum='8efc3abafe0c796eb9aea7b69490d4e2400a33c57504ef4932e1c7105849176f', description=centered.description))
    monkeypatch.setitem(REMOTE_DATASETS, 'bad_checksum', RemoteFileMetadata(name='bad_checksum', filename=filename, url=url, checksum='bad!', description=centered.description))
    UnknownFileMetaData = namedtuple('UnknownFileMetaData', ['filename', 'url', 'checksum', 'description'])
    monkeypatch.setitem(REMOTE_DATASETS, 'test_unknown', UnknownFileMetaData(filename=filename, url=url, checksum='9ae00c83654b3f061d32c882ec0a270d10838fa36515ecb162b89a290e014849', description='Test bad REMOTE_DATASET'))