import datetime
import uuid
from oslo_utils import timeutils
from keystonemiddleware import fixture
from keystonemiddleware.tests.unit.auth_token import test_auth_token_middleware
def test_auth_token_fixture_expired_token(self):
    expired_token_id = uuid.uuid4().hex
    self.atm_fixture.add_token_data(token_id=expired_token_id, user_id=self.user_id, role_list=self.role_list, expires=timeutils.utcnow() - datetime.timedelta(seconds=86400))
    self.call_middleware(headers={'X-Auth-Token': expired_token_id}, expected_status=401)