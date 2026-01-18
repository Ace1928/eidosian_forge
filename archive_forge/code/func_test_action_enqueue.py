import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_enqueue(self, dp):
    port = 207
    queue_id = 4287508753
    self._verify = [dp.ofproto.OFPAT_ENQUEUE, ['port', 'queue_id'], [port, queue_id]]
    action = dp.ofproto_parser.OFPActionEnqueue(port, queue_id)
    self.add_action(dp, [action])