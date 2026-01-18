import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
def test_list_nodes_fills_created_datetime(self):
    nodes = self.driver.list_nodes()
    self.assertEqual(nodes[0].created_at, datetime(2014, 5, 22, 12, 57, 22, 514298, tzinfo=UTC))