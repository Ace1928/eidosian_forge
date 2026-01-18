import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.profiler')
def test_after_execute(self, mock_profiler):
    handler = sqlalchemy._after_cursor_execute()
    handler(mock.MagicMock(), 1, 2, 3, 4, 5)
    mock_profiler.stop.assert_called_once_with()