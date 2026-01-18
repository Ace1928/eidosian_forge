import unittest
from time import time
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_initiation(self):
    initiation_info = [{'type': bmp.BMP_INIT_TYPE_STRING, 'value': 'This is OSKen BGP BMP message'}]
    msg = bmp.BMPInitiation(info=initiation_info)
    binmsg = msg.serialize()
    msg2, rest = bmp.BMPMessage.parser(binmsg)
    self.assertEqual(msg.to_jsondict(lambda v: v), msg2.to_jsondict(lambda v: v))
    self.assertEqual(rest, b'')