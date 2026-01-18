import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def test_ex_find_node(self):
    node = self.driver.ex_find_node('testNode')
    self.assertEqual(node.name, 'testNode')
    node = self.driver.ex_find_node('testNode', self.driver.vdcs[0])
    self.assertEqual(node.name, 'testNode')
    node = self.driver.ex_find_node('testNonExisting', self.driver.vdcs[0])
    self.assertIsNone(node)