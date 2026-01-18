from glance.hacking import checks
from glance.tests import utils
def test_no_contextlib_nested(self):
    self.assertEqual(1, len(list(checks.check_no_contextlib_nested('with contextlib.nested('))))
    self.assertEqual(1, len(list(checks.check_no_contextlib_nested('with nested('))))
    self.assertEqual(0, len(list(checks.check_no_contextlib_nested('with foo as bar'))))