import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_bad_utf8_stringlength(self):
    file_bytes = CONSTANT_ROUTE_CODE[:4] + b'?' + CONSTANT_ROUTE_CODE[5:-4] + b'\xbe)\xe0\xc2'
    self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'UTF8 string at offset 2 extends past end of packet: claimed 63 bytes, 10 available', mime_type='text/plain;charset=utf8')])