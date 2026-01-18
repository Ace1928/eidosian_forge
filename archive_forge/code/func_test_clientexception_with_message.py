from keystoneauth1 import exceptions
from keystoneauth1.tests.unit import utils
def test_clientexception_with_message(self):
    test_message = 'Unittest exception message.'
    exc = exceptions.ClientException(message=test_message)
    self.assertEqual(test_message, exc.message)