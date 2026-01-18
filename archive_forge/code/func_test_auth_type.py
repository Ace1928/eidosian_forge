import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_auth_type(self):
    self.assertIsNone(self.request.auth_type)
    self.request.environ['AUTH_TYPE'] = 'NeGoTiatE'
    self.assertEqual('negotiate', self.request.auth_type)