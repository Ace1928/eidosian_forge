import testtools
from oslotest import base
from octavia_lib.hacking import checks
def test_assert_equal_or_not_none(self):
    self.assertEqual(1, len(list(checks.assert_equal_or_not_none('self.assertEqual(A, None)'))))
    self.assertEqual(1, len(list(checks.assert_equal_or_not_none('self.assertEqual(None, A)'))))
    self.assertEqual(1, len(list(checks.assert_equal_or_not_none('self.assertNotEqual(A, None)'))))
    self.assertEqual(1, len(list(checks.assert_equal_or_not_none('self.assertNotEqual(None, A)'))))
    self.assertEqual(0, len(list(checks.assert_equal_or_not_none('self.assertIsNone()'))))
    self.assertEqual(0, len(list(checks.assert_equal_or_not_none('self.assertIsNotNone()'))))