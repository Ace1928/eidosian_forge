from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
def test_container_get(self):
    self.verify_get(self.proxy.get_container, container.Container)