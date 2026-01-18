from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testHidUsageSelector(self):
    u2f_device = {'usage_page': 61904, 'usage': 1}
    other_device = {'usage_page': 6, 'usage': 1}
    self.assertTrue(hidtransport.HidUsageSelector(u2f_device))
    self.assertFalse(hidtransport.HidUsageSelector(other_device))