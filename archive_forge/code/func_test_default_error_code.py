from os_brick import exception
from os_brick.tests import base
def test_default_error_code(self):

    class FakeBrickException(exception.BrickException):
        code = 404
    exc = FakeBrickException()
    self.assertEqual(exc.kwargs['code'], 404)