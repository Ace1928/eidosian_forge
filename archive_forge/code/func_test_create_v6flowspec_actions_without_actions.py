import logging
import unittest
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
def test_create_v6flowspec_actions_without_actions(self):
    actions = None
    expected_communities = []
    self._test_create_v6flowspec_actions(actions, expected_communities)