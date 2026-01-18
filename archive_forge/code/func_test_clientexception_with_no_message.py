from keystoneauth1 import exceptions
from keystoneauth1.tests.unit import utils
def test_clientexception_with_no_message(self):
    exc = exceptions.ClientException()
    self.assertEqual(exceptions.ClientException.__name__, exc.message)