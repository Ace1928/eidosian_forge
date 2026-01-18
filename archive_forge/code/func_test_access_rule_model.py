from keystone.application_credential.backends import sql as sql_driver
from keystone.common import provider_api
from keystone.common import sql
from keystone.tests.unit.application_credential import test_backends
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
def test_access_rule_model(self):
    cols = (('id', sql.Integer, None), ('external_id', sql.String, 64), ('user_id', sql.String, 64), ('service', sql.String, 64), ('path', sql.String, 128), ('method', sql.String, 16))
    self.assertExpectedSchema('access_rule', cols)