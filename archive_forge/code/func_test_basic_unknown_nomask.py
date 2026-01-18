import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_basic_unknown_nomask(self):
    user = ('field_100', 'aG9nZWhvZ2U=')
    on_wire = b'\x00\x00\xc8\x08hogehoge'
    self._test(user, on_wire, 4)