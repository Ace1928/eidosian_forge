from os_brick import exception
from os_brick.tests import base
def test_error_code_from_kwarg(self):

    class FakeBrickException(exception.BrickException):
        code = 500
    exc = FakeBrickException(code=404)
    self.assertEqual(exc.kwargs['code'], 404)