from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
def test_hosts(self):
    self.verify_list(self.proxy.hosts, host.Host, method_args=[SEGMENT_ID], expected_args=[], expected_kwargs={'segment_id': SEGMENT_ID})