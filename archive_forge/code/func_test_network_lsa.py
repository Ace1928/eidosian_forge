import unittest
from os_ken.lib.packet import ospf
def test_network_lsa(self):
    msg = ospf.NetworkLSA(id_='192.168.0.1', adv_router='192.168.0.2', mask='255.255.255.0', routers=['192.168.0.2'])
    binmsg = msg.serialize()
    msg2, cls, rest = ospf.LSA.parser(binmsg)
    self.assertEqual(msg.header.checksum, msg2.header.checksum)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(rest, b'')