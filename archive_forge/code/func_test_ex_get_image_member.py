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
def test_ex_get_image_member(self):
    image_id = 'd9a9cd9a-278a-444c-90a6-d24b8c688a63'
    image_member_id = '016926dff12345e8b10329f24c99745b'
    image_member = self.driver.ex_get_image_member(image_id, image_member_id)
    self.assertEqual(image_member.id, image_member_id)
    self.assertEqual(image_member.image_id, image_id)
    self.assertEqual(image_member.state, NodeImageMemberState.ACCEPTED)
    self.assertEqual(image_member.created, '2017-01-12T12:31:50Z')
    self.assertEqual(image_member.extra['updated'], '2017-01-12T12:31:54Z')
    self.assertEqual(image_member.extra['schema'], '/v2/schemas/member')