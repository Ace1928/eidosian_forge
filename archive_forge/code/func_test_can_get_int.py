from tests.compat import mock, unittest
from boto.pyami import config
from boto.compat import StringIO
def test_can_get_int(self):
    self.assertEqual(self.config.getint('Boto', 'http_socket_timeout'), 1)
    self.assertEqual(self.config.getint('Boto', 'does-not-exist'), 0)
    self.assertEqual(self.config.getint('Boto', 'does-not-exist', default=20), 20)