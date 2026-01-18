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
def test_manage_router_interfaces(self):
    router = self.driver.ex_list_routers()[1]
    port = self.driver.ex_list_ports()[0]
    subnet = self.driver.ex_list_subnets()[0]
    self.assertTrue(self.driver.ex_add_router_port(router, port))
    self.assertTrue(self.driver.ex_del_router_port(router, port))
    self.assertTrue(self.driver.ex_add_router_subnet(router, subnet))
    self.assertTrue(self.driver.ex_del_router_subnet(router, subnet))