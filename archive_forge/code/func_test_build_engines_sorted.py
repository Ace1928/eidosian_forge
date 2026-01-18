from __future__ import annotations
import sys
from importlib.metadata import EntryPoint
from unittest import mock
import pytest
from xarray.backends import common, plugins
from xarray.tests import (
importlib_metadata_mock = "importlib.metadata"
@mock.patch(f'{importlib_metadata_mock}.EntryPoint.load', mock.MagicMock(return_value=DummyBackendEntrypoint1))
def test_build_engines_sorted() -> None:
    dummy_pkg_entrypoints = EntryPoints([EntryPoint('dummy2', 'xarray.tests.test_plugins:backend_1', 'xarray.backends'), EntryPoint('dummy1', 'xarray.tests.test_plugins:backend_1', 'xarray.backends')])
    backend_entrypoints = list(plugins.build_engines(dummy_pkg_entrypoints))
    indices = []
    for be in plugins.STANDARD_BACKENDS_ORDER:
        try:
            index = backend_entrypoints.index(be)
            backend_entrypoints.pop(index)
            indices.append(index)
        except ValueError:
            pass
    assert set(indices) < {0, -1}
    assert list(backend_entrypoints) == sorted(backend_entrypoints)