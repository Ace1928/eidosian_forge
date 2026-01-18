import logging
import os
import sys
import unittest
from os_ken.lib import pcaplib
from os_ken.lib.packet import gre
from os_ken.lib.packet import packet
from os_ken.utils import binary_str
from os_ken.lib.packet.ether_types import ETH_TYPE_IP, ETH_TYPE_TEB
def test_key_setter(self):
    self.gre.key = self.key
    self.assertEqual(self.gre._key, self.key)
    self.assertEqual(self.gre._vsid, self.vsid)
    self.assertEqual(self.gre._flow_id, self.flow_id)