import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.lib import addrconv
from struct import unpack
def check_old(o):

    def get_field(m, t):
        for f in m.fields:
            if isinstance(f, t):
                return f
    get_value = lambda m, t: get_field(m, t).value
    self.assertEqual(get_value(o, ofpp.MTInPort), old_in_port)
    self.assertEqual(get_value(o, ofpp.MTEthSrc), old_eth_src)
    self.assertEqual(get_value(o, ofpp.MTIPV4Src), old_ipv4_src)
    self.assertEqual(get_value(o, ofpp.MTIPv6Src), old_ipv6_src)