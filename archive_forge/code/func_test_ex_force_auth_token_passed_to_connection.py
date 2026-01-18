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
def test_ex_force_auth_token_passed_to_connection(self):
    base_url = 'https://servers.api.rackspacecloud.com/v1.1/slug'
    kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_token': 'preset-auth-token', 'ex_force_auth_url': 'https://auth.api.example.com', 'ex_force_base_url': base_url}
    driver = self.driver_type(*self.driver_args, **kwargs)
    driver.list_nodes()
    self.assertEqual(kwargs['ex_force_auth_token'], driver.connection.auth_token)
    self.assertEqual('servers.api.rackspacecloud.com', driver.connection.host)
    self.assertEqual('/v1.1/slug', driver.connection.request_path)
    self.assertEqual(443, driver.connection.port)