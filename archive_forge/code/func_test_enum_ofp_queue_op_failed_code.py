import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_queue_op_failed_code(self):
    self.assertEqual(OFPQOFC_BAD_PORT, 0)
    self.assertEqual(OFPQOFC_BAD_QUEUE, 1)
    self.assertEqual(OFPQOFC_EPERM, 2)