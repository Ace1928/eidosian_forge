from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_find_bgp_speaker(self):
    sot = self.operator_cloud.network.find_bgp_speaker(self.SPEAKER.name)
    self.assertEqual(self.IP_VERSION, sot.ip_version)
    self.assertEqual(self.LOCAL_AS, sot.local_as)
    self.assertTrue(sot.advertise_floating_ip_host_routes)
    self.assertTrue(sot.advertise_tenant_networks)