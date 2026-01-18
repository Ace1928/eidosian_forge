from openstack.database.v1 import _proxy
from openstack.database.v1 import database
from openstack.database.v1 import flavor
from openstack.database.v1 import instance
from openstack.database.v1 import user
from openstack.tests.unit import test_proxy_base
def test_instance_update(self):
    self.verify_update(self.proxy.update_instance, instance.Instance)