import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
@pytest.fixture
def jp_server_config(capsys):
    return {'ServerApp': {'jpserver_extensions': {'notebook_shim': True, 'notebook_shim.tests.mockextension': True}}}