import copy
import logging
from struct import pack, unpack_from
import unittest
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib import addrconv
from os_ken.lib.packet.slow import slow, lacp
from os_ken.lib.packet.slow import SLOW_PROTOCOL_MULTICAST
from os_ken.lib.packet.slow import SLOW_SUBTYPE_LACP
from os_ken.lib.packet.slow import SLOW_SUBTYPE_MARKER
def test_invalid_actor_tag(self):
    self.skipTest('This test needs to be refactored: "serialize" method requires two input parameters that are not provided here.')
    invalid_lacv = copy.deepcopy(self.l)
    invalid_lacv.actor_tag = 4
    invalid_buf = invalid_lacv.serialize()
    slow.parser(invalid_buf)