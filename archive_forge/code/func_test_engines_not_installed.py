from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
@mock.patch('xarray.backends.plugins.list_engines', mock.MagicMock(return_value={}))
def test_engines_not_installed() -> None:
    with pytest.raises(ValueError, match='xarray is unable to open'):
        plugins.guess_engine('not-valid')
    with pytest.raises(ValueError, match='found the following matches with the input'):
        plugins.guess_engine('foo.nc')