import unittest
from os_ken.lib.packet import ospf
def test_lsupd(self):
    link1 = ospf.RouterLSA.Link(id_='10.0.0.1', data='255.255.255.0', type_=ospf.LSA_LINK_TYPE_STUB, metric=10)
    lsa1 = ospf.RouterLSA(id_='192.168.0.1', adv_router='192.168.0.2', links=[link1])
    msg = ospf.OSPFLSUpd(router_id='192.168.0.1', lsas=[lsa1])
    binmsg = msg.serialize()
    msg2, cls, rest = ospf.OSPFMessage.parser(binmsg)
    self.assertEqual(msg.checksum, msg2.checksum)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(rest, b'')