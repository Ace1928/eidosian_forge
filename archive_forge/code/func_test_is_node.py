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
def test_is_node(self):
    self.assertTrue(self.driver._is_node(Node('testId', 'testNode', state=0, public_ips=[], private_ips=[], driver=self.driver)))
    self.assertFalse(self.driver._is_node(NodeImage('testId', 'testNode', driver=self.driver)))