import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def check_events(self, source_bytes, events):
    source = BytesIO(source_bytes)
    result = StreamResult()
    subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
    self.assertEqual(b'', source.read())
    self.assertEqual(events, result._events)
    for event in result._events:
        if event[5] is not None:
            bytes(event[6])