import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib import addrconv
def test_json_with_auth_simple(self):
    auth_cls = bfd.SimplePassword(auth_key_id=2, password=self.auth_keys[2])
    bfd1 = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
    jsondict = bfd1.to_jsondict()
    bfd2 = bfd.bfd.from_jsondict(jsondict['bfd'])
    self.assertEqual(str(bfd1), str(bfd2))