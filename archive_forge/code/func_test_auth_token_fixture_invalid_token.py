import datetime
import uuid
from oslo_utils import timeutils
from keystonemiddleware import fixture
from keystonemiddleware.tests.unit.auth_token import test_auth_token_middleware
def test_auth_token_fixture_invalid_token(self):
    self.call_middleware(headers={'X-Auth-Token': uuid.uuid4().hex}, expected_status=401)