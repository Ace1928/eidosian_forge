import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_stats_reply_flags(self):
    self.assertEqual(OFPSF_REPLY_MORE, 1)