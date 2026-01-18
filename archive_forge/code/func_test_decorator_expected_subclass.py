import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_decorator_expected_subclass(self):

    class FooException(Exception):
        pass

    class BarException(FooException):
        pass

    @oslo_messaging.expected_exceptions(FooException)
    def naughty():
        raise BarException()
    self.assertRaises(oslo_messaging.ExpectedException, naughty)