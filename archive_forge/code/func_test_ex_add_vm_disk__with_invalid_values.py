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
def test_ex_add_vm_disk__with_invalid_values(self):
    self.assertRaises(ValueError, self.driver.ex_add_vm_disk, 'dummy', 'invalid value')
    self.assertRaises(ValueError, self.driver.ex_add_vm_disk, 'dummy', '-1')