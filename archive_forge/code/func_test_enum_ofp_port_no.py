import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_no(self):
    self.assertEqual(OFPP_MAX, 4294967040)
    self.assertEqual(OFPP_IN_PORT, 4294967288)
    self.assertEqual(OFPP_TABLE, 4294967289)
    self.assertEqual(OFPP_NORMAL, 4294967290)
    self.assertEqual(OFPP_FLOOD, 4294967291)
    self.assertEqual(OFPP_ALL, 4294967292)
    self.assertEqual(OFPP_CONTROLLER, 4294967293)
    self.assertEqual(OFPP_LOCAL, 4294967294)
    self.assertEqual(OFPP_ANY, 4294967295)
    self.assertEqual(OFPQ_ALL, 4294967295)