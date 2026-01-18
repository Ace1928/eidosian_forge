from tests.compat import mock, unittest
from boto.pyami import config
from boto.compat import StringIO
def test_can_get_bool(self):
    self.assertTrue(self.config.getbool('Boto', 'https_validate_certificates'))
    self.assertFalse(self.config.getbool('Boto', 'other'))
    self.assertFalse(self.config.getbool('Boto', 'does-not-exist'))