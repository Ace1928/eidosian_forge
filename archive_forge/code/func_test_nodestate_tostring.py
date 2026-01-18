import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_nodestate_tostring(self):
    self.assertEqual(NodeState.tostring(NodeState.RUNNING), 'RUNNING')