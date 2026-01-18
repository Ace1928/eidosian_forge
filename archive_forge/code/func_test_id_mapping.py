import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def test_id_mapping(self):
    cols = (('public_id', sql.String, 64), ('domain_id', sql.String, 64), ('local_id', sql.String, 255), ('entity_type', sql.Enum, None))
    self.assertExpectedSchema('id_mapping', cols)