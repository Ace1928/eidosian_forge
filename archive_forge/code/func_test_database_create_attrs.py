from openstack.database.v1 import _proxy
from openstack.database.v1 import database
from openstack.database.v1 import flavor
from openstack.database.v1 import instance
from openstack.database.v1 import user
from openstack.tests.unit import test_proxy_base
def test_database_create_attrs(self):
    self.verify_create(self.proxy.create_database, database.Database, method_kwargs={'instance': 'id'}, expected_kwargs={'instance_id': 'id'})