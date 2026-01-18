from novaclient import exceptions
from novaclient.tests.unit import utils as test_utils
def test_from_response_webob_post_1_6_0(self):
    message = 'Flavor test could not be found.'
    self._test_from_response({'message': message, 'code': 404}, message)