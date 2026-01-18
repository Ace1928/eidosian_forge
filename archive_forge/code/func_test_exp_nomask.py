import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_nomask(self):
    user = ('_dp_hash', 305419896)
    on_wire = b'\xff\xff\x00\x08\x00\x00# \x124Vx'
    self._test(user, on_wire, 8)