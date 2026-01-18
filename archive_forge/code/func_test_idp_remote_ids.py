from keystone.common import sql
from keystone.tests.unit import test_backend_sql
def test_idp_remote_ids(self):
    cols = (('idp_id', sql.String, 64), ('remote_id', sql.String, 255))
    self.assertExpectedSchema('idp_remote_ids', cols)