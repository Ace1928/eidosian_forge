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
def test_ex_get_node_security_groups(self):
    node = Node(id='1c01300f-ef97-4937-8f03-ac676d6234be', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
    security_groups = self.driver.ex_get_node_security_groups(node)
    self.assertEqual(len(security_groups), 2, 'Wrong security groups count')
    security_group = security_groups[1]
    self.assertEqual(security_group.id, 4)
    self.assertEqual(security_group.tenant_id, '68')
    self.assertEqual(security_group.name, 'ftp')
    self.assertEqual(security_group.description, 'FTP Client-Server - Open 20-21 ports')
    self.assertEqual(security_group.rules[0].id, 1)
    self.assertEqual(security_group.rules[0].parent_group_id, 4)
    self.assertEqual(security_group.rules[0].ip_protocol, 'tcp')
    self.assertEqual(security_group.rules[0].from_port, 20)
    self.assertEqual(security_group.rules[0].to_port, 21)
    self.assertEqual(security_group.rules[0].ip_range, '0.0.0.0/0')