from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from openstack.tests.functional import base
def test_list_bgp_speakers(self):
    speaker_ids = [sp.id for sp in self.operator_cloud.network.bgp_speakers()]
    self.assertIn(self.SPEAKER.id, speaker_ids)