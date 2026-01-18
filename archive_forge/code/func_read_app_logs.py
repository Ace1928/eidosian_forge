import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
@pytest.fixture
def read_app_logs(capsys):
    """Fixture that returns a callable to read
    the current output from the application's logs
    that was printed to sys.stderr.
    """

    def _inner():
        captured = capsys.readouterr()
        return captured.err
    return _inner