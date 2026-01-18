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
def test_ex_create_snapshot_does_not_post_optional_parameters_if_none(self):
    volume = self.driver.list_volumes()[0]
    with patch.object(self.driver, '_to_snapshot'):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume_snapshot(volume, name=None, ex_description=None, ex_force=True)
    name, args, kwargs = mock_request.mock_calls[0]
    self.assertFalse('display_name' in kwargs['data']['snapshot'])
    self.assertFalse('display_description' in kwargs['data']['snapshot'])