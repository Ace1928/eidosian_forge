import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_decorator_expected(self):

    class FooException(Exception):
        pass

    @oslo_messaging.expected_exceptions(FooException)
    def naughty():
        raise FooException()
    self.assertRaises(oslo_messaging.ExpectedException, naughty)