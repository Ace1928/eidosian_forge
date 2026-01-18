import logging
import unittest
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
def test_create_l2vpnflowspec_actions_not_exist_actions(self):
    actions = {'traffic_test': {'test': 10}}
    expected_communities = []
    self.assertRaises(ValueError, self._test_create_l2vpnflowspec_actions, actions, expected_communities)