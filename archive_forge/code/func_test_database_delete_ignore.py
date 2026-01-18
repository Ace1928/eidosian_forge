from openstack.database.v1 import _proxy
from openstack.database.v1 import database
from openstack.database.v1 import flavor
from openstack.database.v1 import instance
from openstack.database.v1 import user
from openstack.tests.unit import test_proxy_base
def test_database_delete_ignore(self):
    self.verify_delete(self.proxy.delete_database, database.Database, ignore_missing=True, method_kwargs={'instance': 'test_id'}, expected_kwargs={'instance_id': 'test_id'})