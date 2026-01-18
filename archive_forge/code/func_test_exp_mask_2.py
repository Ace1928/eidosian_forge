import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_mask_2(self):
    user = ('tcp_flags', (2166, 2047))
    on_wire = b'\xff\xffU\x08ONF\x00\x08v\x07\xff'
    self._test(user, on_wire, 8)