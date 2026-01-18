import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_ext_256_mask(self):
    user = ('pbb_uca', (50, 51))
    on_wire = b'\xff\xff\x01\x08ONF\x00\n\x0023'
    self._test(user, on_wire, 10)