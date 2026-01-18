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
def test_backend_credential_sql_encrypted_string(self):
    cred_dict = {'id': uuid.uuid4().hex, 'type': uuid.uuid4().hex, 'hash': uuid.uuid4().hex, 'encrypted_blob': b'randomdata'}
    ref = credential_sql.CredentialModel.from_dict(cred_dict)
    self.assertIsInstance(ref.encrypted_blob, str)