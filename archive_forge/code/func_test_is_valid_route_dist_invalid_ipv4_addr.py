import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_route_dist_invalid_ipv4_addr(self):
    self.assertEqual(False, validation.is_valid_route_dist('xxx.xxx.xxx.xxx:333'))