import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_ext_256_nomask(self):
    user = ('pbb_uca', 50)
    on_wire = b'\xff\xff\x00\x07ONF\x00\n\x002'
    self._test(user, on_wire, 10)