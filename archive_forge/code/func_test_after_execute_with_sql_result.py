import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.profiler')
def test_after_execute_with_sql_result(self, mock_profiler):
    handler = sqlalchemy._after_cursor_execute(hide_result=False)
    cursor = mock.MagicMock()
    cursor._rows = (1,)
    handler(1, cursor, 2, 3, 4, 5)
    info = {'db': {'result': str(cursor._rows)}}
    mock_profiler.stop.assert_called_once_with(info=info)