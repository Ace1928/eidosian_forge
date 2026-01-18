import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.lib import addrconv
from struct import unpack
def check_new(o):
    self.assertEqual(o['in_port'], in_port)
    self.assertEqual(o['eth_src'], eth_src)
    self.assertEqual(o['ipv4_src'], ipv4_src)
    self.assertEqual(o['ipv6_src'], ipv6_src)