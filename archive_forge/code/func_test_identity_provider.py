from keystone.common import sql
from keystone.tests.unit import test_backend_sql
def test_identity_provider(self):
    cols = (('id', sql.String, 64), ('domain_id', sql.String, 64), ('enabled', sql.Boolean, None), ('description', sql.Text, None), ('authorization_ttl', sql.Integer, None))
    self.assertExpectedSchema('identity_provider', cols)