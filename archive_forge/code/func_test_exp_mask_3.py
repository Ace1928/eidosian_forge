import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_mask_3(self):
    user = ('actset_output', (2557891634, 4294967294))
    on_wire = b'\xff\xffW\x0cONF\x00\x98vT2\xff\xff\xff\xfe'
    self._test(user, on_wire, 8)