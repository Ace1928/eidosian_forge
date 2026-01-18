import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_list_users_by_password_expires_with_bad_timestamp_fails(self):
    """Ensure an invalid timestamp returns a Bad Request.

        GET /users?password_expires_at={invalid_timestamp}
        GET /users?password_expires_at={operator}:{timestamp}&
        {operator}:{invalid_timestamp}

        """
    bad_ts_url = self._list_users_by_password_expires_at(self.starttime.strftime('%S:%M:%ST%Y-%m-%d'))
    self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)
    bad_ts_url = self._list_users_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self.starttime.strftime('%S:%M:%ST%Y-%m-%d'), 'gt')
    self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)