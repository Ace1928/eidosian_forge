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
def test_ex_create_security_group_rule(self):
    security_group = OpenStackSecurityGroup(id=6, tenant_id=None, name=None, description=None, driver=self.driver)
    security_group_rule = self.driver.ex_create_security_group_rule(security_group, 'tcp', 14, 16, '0.0.0.0/0')
    self.assertEqual(security_group_rule.id, 2)
    self.assertEqual(security_group_rule.parent_group_id, 6)
    self.assertEqual(security_group_rule.ip_protocol, 'tcp')
    self.assertEqual(security_group_rule.from_port, 14)
    self.assertEqual(security_group_rule.to_port, 16)
    self.assertEqual(security_group_rule.ip_range, '0.0.0.0/0')
    self.assertIsNone(security_group_rule.tenant_id)