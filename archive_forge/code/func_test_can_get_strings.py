from tests.compat import mock, unittest
from boto.pyami import config
from boto.compat import StringIO
def test_can_get_strings(self):
    self.assertEqual(self.config.get('Credentials', 'aws_access_key_id'), 'foo')
    self.assertIsNone(self.config.get('Credentials', 'no-exist'))
    self.assertEqual(self.config.get('Credentials', 'no-exist', 'default-value'), 'default-value')