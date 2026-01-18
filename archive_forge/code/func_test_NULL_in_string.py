import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_NULL_in_string(self):
    file_bytes = CONSTANT_ROUTE_CODE[:6] + b'\x00' + CONSTANT_ROUTE_CODE[7:-4] + b'\xd7A\xac\xfe'
    self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'UTF8 string at offset 2 contains NUL byte', mime_type='text/plain;charset=utf8')])