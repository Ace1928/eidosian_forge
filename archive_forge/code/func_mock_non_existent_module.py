import os
import sys
import mock
import pytest  # type: ignore
@pytest.fixture
def mock_non_existent_module(monkeypatch):
    """Mocks a non-existing module in sys.modules.

    Additionally mocks any non-existing modules specified in the dotted path.
    """

    def _mock_non_existent_module(path):
        parts = path.split('.')
        partial = []
        for part in parts:
            partial.append(part)
            current_module = '.'.join(partial)
            if current_module not in sys.modules:
                monkeypatch.setitem(sys.modules, current_module, mock.MagicMock())
    return _mock_non_existent_module