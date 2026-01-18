from os_brick import exception
from os_brick.tests import base
def test_error_msg(self):
    self.assertEqual(str(exception.BrickException('test')), 'test')