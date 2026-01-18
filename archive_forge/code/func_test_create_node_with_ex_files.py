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
def test_create_node_with_ex_files(self):
    OpenStack_2_0_MockHttp.type = 'EX_FILES'
    image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
    size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
    files = {'/file1': 'content1', '/file2': 'content2'}
    node = self.driver.create_node(name='racktest', image=image, size=size, ex_files=files)
    self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
    self.assertEqual(node.name, 'racktest')
    OpenStack_2_0_MockHttp.type = 'EX_FILES_NONE'
    node = self.driver.create_node(name='racktest', image=image, size=size)
    self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
    self.assertEqual(node.name, 'racktest')