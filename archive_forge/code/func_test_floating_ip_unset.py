from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_floating_ip_unset(self):
    self._verify('openstack.proxy.Proxy._update', self.proxy.unset_floating_ip, method_args=['value'], method_kwargs={}, expected_args=[floating_ip.FloatingIP, 'value'], expected_kwargs={'ptrdname': None})