import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_node_in_maintenace_state(self):
    self.mock_operations.get_node_state.side_effect = ['maintenance', 'maintenance', None]
    self.assertTrue(self.destroyer.destroy_node(1))
    self.mock_sleep.assert_has_calls([call(self.destroyer.WAIT_AMOUNT), call(self.destroyer.WAIT_AMOUNT)])
    self.assertTrue(self.mock_operations.stop_node.call_count == 0)
    self.assertTrue(self.mock_operations.destroy_node.call_count == 0)