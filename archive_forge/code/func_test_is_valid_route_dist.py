import logging
import unittest
from os_ken.services.protocols.bgp.utils import validation
def test_is_valid_route_dist(self):
    self.assertTrue(validation.is_valid_route_dist('65000:222'))