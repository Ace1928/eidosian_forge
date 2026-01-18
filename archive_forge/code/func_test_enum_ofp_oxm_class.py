import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_oxm_class(self):
    self.assertEqual(OFPXMC_NXM_0, 0)
    self.assertEqual(OFPXMC_NXM_1, 1)
    self.assertEqual(OFPXMC_OPENFLOW_BASIC, 32768)
    self.assertEqual(OFPXMC_EXPERIMENTER, 65535)