from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_get_bgp_peer(self):
    sot = self.operator_cloud.network.get_bgp_peer(self.PEER.id)
    self.assertEqual(self.PEER_IP, sot.peer_ip)
    self.assertEqual(self.REMOTE_AS, sot.remote_as)