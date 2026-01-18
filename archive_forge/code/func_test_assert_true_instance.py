from glance.hacking import checks
from glance.tests import utils
def test_assert_true_instance(self):
    self.assertEqual(1, len(list(checks.assert_true_instance('self.assertTrue(isinstance(e, exception.BuildAbortException))'))))
    self.assertEqual(0, len(list(checks.assert_true_instance('self.assertTrue()'))))