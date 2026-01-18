import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
def test_backend_credential_sql_no_hints(self):
    credentials = PROVIDERS.credential_api.list_credentials()
    self._validate_credential_list(credentials, self.user_credentials)