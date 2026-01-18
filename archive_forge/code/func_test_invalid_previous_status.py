from heat.engine import support
from heat.tests import common
def test_invalid_previous_status(self):
    ex = self.assertRaises(ValueError, support.SupportStatus, previous_status='YARRR')
    self.assertEqual('previous_status must be SupportStatus instead of %s' % str, str(ex))