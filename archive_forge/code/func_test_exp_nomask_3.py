import unittest
import os_ken.ofproto.ofproto_v1_3 as ofp
def test_exp_nomask_3(self):
    user = ('actset_output', 2557891634)
    on_wire = b'\xff\xffV\x08ONF\x00\x98vT2'
    self._test(user, on_wire, 8)