from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
def test_host_get(self):
    self.verify_get(self.proxy.get_host, host.Host, method_args=[HOST_ID], method_kwargs={'segment_id': SEGMENT_ID}, expected_kwargs={'segment_id': SEGMENT_ID})