import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def test_basic_single(self):
    user = ('flow_count', 100)
    on_wire = b'\x80\x02\x06\x04\x00\x00\x00d'
    self._test(user, on_wire, 4)