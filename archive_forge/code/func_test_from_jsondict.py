import logging
import unittest
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vxlan
def test_from_jsondict(self):
    pkt_from_json = vxlan.vxlan.from_jsondict(self.jsondict[vxlan.vxlan.__name__])
    self.assertEqual(self.vni, pkt_from_json.vni)