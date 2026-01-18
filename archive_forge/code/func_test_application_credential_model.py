from keystone.application_credential.backends import sql as sql_driver
from keystone.common import provider_api
from keystone.common import sql
from keystone.tests.unit.application_credential import test_backends
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
def test_application_credential_model(self):
    cols = (('internal_id', sql.Integer, None), ('id', sql.String, 64), ('name', sql.String, 255), ('secret_hash', sql.String, 255), ('description', sql.Text, None), ('user_id', sql.String, 64), ('project_id', sql.String, 64), ('system', sql.String, 64), ('expires_at', sql.DateTimeInt, None))
    self.assertExpectedSchema('application_credential', cols)