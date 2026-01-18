import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_decorator_unexpected(self):

    class FooException(Exception):
        pass

    @oslo_messaging.expected_exceptions(FooException)
    def really_naughty():
        raise ValueError()
    self.assertRaises(ValueError, really_naughty)