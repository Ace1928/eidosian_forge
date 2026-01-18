import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_node_in_error_state(self):
    self.mock_operations.get_node_state.side_effect = ['error']
    self.assertFalse(self.destroyer.destroy_node(1))
    self.assertTrue(self.mock_operations.stop_node.call_count == 0)
    self.assertTrue(self.mock_operations.destroy_node.call_count == 0)