from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
@mock.patch(f'{importlib_metadata_mock}.EntryPoint.load', mock.MagicMock(return_value=None))
def test_backends_dict_from_pkg() -> None:
    specs = [['engine1', 'xarray.tests.test_plugins:backend_1', 'xarray.backends'], ['engine2', 'xarray.tests.test_plugins:backend_2', 'xarray.backends']]
    entrypoints = [EntryPoint(name, value, group) for name, value, group in specs]
    engines = plugins.backends_dict_from_pkg(entrypoints)
    assert len(engines) == 2
    assert engines.keys() == {'engine1', 'engine2'}