import unittest
import logging
from os_ken.ofproto.ether import *
def test_ether_type(self):
    self.assertEqual(ETH_TYPE_IP, 2048)
    self.assertEqual(ETH_TYPE_ARP, 2054)
    self.assertEqual(ETH_TYPE_8021Q, 33024)
    self.assertEqual(ETH_TYPE_IPV6, 34525)
    self.assertEqual(ETH_TYPE_MPLS, 34887)
    self.assertEqual(ETH_TYPE_SLOW, 34825)