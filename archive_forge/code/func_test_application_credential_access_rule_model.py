from keystone.application_credential.backends import sql as sql_driver
from keystone.common import provider_api
from keystone.common import sql
from keystone.tests.unit.application_credential import test_backends
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
def test_application_credential_access_rule_model(self):
    cols = (('application_credential_id', sql.Integer, None), ('access_rule_id', sql.Integer, None))
    self.assertExpectedSchema('application_credential_access_rule', cols)