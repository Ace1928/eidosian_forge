import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import model_base
from neutron_lib.tests.unit.db import _base as db_base
def test_get_set_tenant_id_project(self):
    foo = TestTable(project_id='project')
    self.assertEqual('project', foo.get_tenant_id())
    foo.set_tenant_id('tenant')
    self.assertEqual('tenant', foo.get_tenant_id())