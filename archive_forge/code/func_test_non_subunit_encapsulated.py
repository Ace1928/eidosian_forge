import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_non_subunit_encapsulated(self):
    source = BytesIO(b'foo\nbar\n')
    result = StreamResult()
    subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
    self.assertEqual([('status', None, None, None, True, 'stdout', b'f', False, None, None, None), ('status', None, None, None, True, 'stdout', b'o', False, None, None, None), ('status', None, None, None, True, 'stdout', b'o', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\n', False, None, None, None), ('status', None, None, None, True, 'stdout', b'b', False, None, None, None), ('status', None, None, None, True, 'stdout', b'a', False, None, None, None), ('status', None, None, None, True, 'stdout', b'r', False, None, None, None), ('status', None, None, None, True, 'stdout', b'\n', False, None, None, None)], result._events)
    self.assertEqual(b'', source.read())