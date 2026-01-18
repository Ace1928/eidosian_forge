import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_dl_dst(self, dp):
    dl_dst = 'c2:93:a2:fb:d0:f4'
    dl_dst_bin = self.haddr_to_bin(dl_dst)
    self._verify = [dp.ofproto.OFPAT_SET_DL_DST, 'dl_addr', dl_dst_bin]
    action = dp.ofproto_parser.OFPActionSetDlDst(dl_dst_bin)
    self.add_action(dp, [action])