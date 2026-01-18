import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_json1(self):
    opt_param = [bgp.BGPOptParamCapabilityUnknown(cap_code=200, cap_value=b'hoge'), bgp.BGPOptParamCapabilityRouteRefresh(), bgp.BGPOptParamCapabilityMultiprotocol(afi=afi.IP, safi=safi.MPLS_VPN), bgp.BGPOptParamCapabilityFourOctetAsNumber(as_number=1234567), bgp.BGPOptParamUnknown(type_=99, value=b'fuga')]
    msg1 = bgp.BGPOpen(my_as=30000, bgp_identifier='192.0.2.2', opt_param=opt_param)
    jsondict = msg1.to_jsondict()
    msg2 = bgp.BGPOpen.from_jsondict(jsondict['BGPOpen'])
    self.assertEqual(str(msg1), str(msg2))