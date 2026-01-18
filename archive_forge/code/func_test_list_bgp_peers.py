from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_list_bgp_peers(self):
    peer_ids = [pe.id for pe in self.operator_cloud.network.bgp_peers()]
    self.assertIn(self.PEER.id, peer_ids)