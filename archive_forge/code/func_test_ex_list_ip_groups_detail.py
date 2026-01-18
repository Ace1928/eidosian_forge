import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
def test_ex_list_ip_groups_detail(self):
    ret = self.driver.ex_list_ip_groups(details=True)
    self.assertEqual(2, len(ret))
    self.assertEqual('1234', ret[0].id)
    self.assertEqual('Shared IP Group 1', ret[0].name)
    self.assertEqual(2, len(ret[0].servers))
    self.assertEqual('422', ret[0].servers[0])
    self.assertEqual('3445', ret[0].servers[1])
    self.assertEqual('5678', ret[1].id)
    self.assertEqual('Shared IP Group 2', ret[1].name)
    self.assertEqual(3, len(ret[1].servers))
    self.assertEqual('23203', ret[1].servers[0])
    self.assertEqual('2456', ret[1].servers[1])
    self.assertEqual('9891', ret[1].servers[2])