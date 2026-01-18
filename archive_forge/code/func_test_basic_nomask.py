import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_basic_nomask(self):
    user = ('ipv4_src', '192.0.2.1')
    on_wire = b'\x80\x00\x16\x04\xc0\x00\x02\x01'
    self._test(user, on_wire, 4)