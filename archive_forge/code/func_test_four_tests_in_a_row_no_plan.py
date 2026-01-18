from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_four_tests_in_a_row_no_plan(self):
    self.tap.write(_u('ok 1 - first test in a script with no plan at all\n'))
    self.tap.write(_u('not ok 2 - second\n'))
    self.tap.write(_u('ok 3 - third\n'))
    self.tap.write(_u('not ok 4 - fourth\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1 - first test in a script with no plan at all', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2 - second', 'fail', None, False, None, None, True, None, None, None), ('status', 'test 3 - third', 'success', None, False, None, None, True, None, None, None), ('status', 'test 4 - fourth', 'fail', None, False, None, None, True, None, None, None)])