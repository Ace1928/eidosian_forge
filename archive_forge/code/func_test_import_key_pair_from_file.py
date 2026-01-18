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
def test_import_key_pair_from_file(self):
    name = 'key3'
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa.pub')
    with open(path) as fp:
        pub_key = fp.read()
    keypair = self.driver.import_key_pair_from_file(name=name, key_file_path=path)
    self.assertEqual(keypair.name, name)
    self.assertEqual(keypair.fingerprint, '97:10:a6:e7:92:65:7e:69:fe:e6:81:8f:39:3c:8f:5a')
    self.assertEqual(keypair.public_key, pub_key)
    self.assertIsNone(keypair.private_key)