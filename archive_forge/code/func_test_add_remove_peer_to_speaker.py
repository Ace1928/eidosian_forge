from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_add_remove_peer_to_speaker(self):
    self.operator_cloud.network.add_bgp_peer_to_speaker(self.SPEAKER.id, self.PEER.id)
    sot = self.operator_cloud.network.get_bgp_speaker(self.SPEAKER.id)
    self.assertEqual([self.PEER.id], sot.peers)
    self.operator_cloud.network.remove_bgp_peer_from_speaker(self.SPEAKER.id, self.PEER.id)
    sot = self.operator_cloud.network.get_bgp_speaker(self.SPEAKER.id)
    self.assertEqual([], sot.peers)