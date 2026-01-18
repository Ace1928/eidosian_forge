from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_missing_test_with_plan_adds_error(self):
    self.tap.write(_u('1..3\n'))
    self.tap.write(_u('ok first test\n'))
    self.tap.write(_u('not ok 3 third test\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1 first test', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2', 'fail', None, False, 'tap meta', b'test missing from TAP output', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 3 third test', 'fail', None, False, None, None, True, None, None, None)])