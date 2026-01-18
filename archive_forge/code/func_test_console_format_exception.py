import io
from oslo_log.cmds import convert_json
from oslo_serialization import jsonutils
from oslotest import base as test_base
def test_console_format_exception(self):
    lines = self._lines(EXCEPTION_RECORD, traceback_key='exception')
    self.assertEqual(['pre msg', 'pre abc', 'pre def'], lines)