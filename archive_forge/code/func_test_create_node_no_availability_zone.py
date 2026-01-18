import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_create_node_no_availability_zone(self):
    image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    node = self.driver.create_node(name='foo', image=image, size=size)
    location = NodeLocation(0, 'Amazon US N. Virginia', 'US', self.driver)
    self.assertEqual(node.id, 'i-2ba64342')
    node = self.driver.create_node(name='foo', image=image, size=size, location=location)
    self.assertEqual(node.id, 'i-2ba64342')
    self.assertEqual(node.name, 'foo')