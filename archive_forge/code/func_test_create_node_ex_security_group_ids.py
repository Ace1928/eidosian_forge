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
def test_create_node_ex_security_group_ids(self):
    EC2MockHttp.type = 'ex_security_group_ids'
    image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    subnet = EC2NetworkSubnet(12345, 'test_subnet', 'pending')
    security_groups = ['sg-1aa11a1a', 'sg-2bb22b2b']
    self.driver.create_node(name='foo', image=image, size=size, ex_security_group_ids=security_groups, ex_subnet=subnet)
    self.assertRaises(ValueError, self.driver.create_node, name='foo', image=image, size=size, ex_security_group_ids=security_groups)