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
def test_create_node_with_ex_assign_public_ip(self):
    EC2MockHttp.type = 'create_ex_assign_public_ip'
    image = NodeImage(id='ami-11111111', name=self.image_name, driver=self.driver)
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    subnet = EC2NetworkSubnet('subnet-11111111', 'test_subnet', 'pending')
    self.driver.create_node(name='foo', image=image, size=size, ex_subnet=subnet, ex_security_group_ids=['sg-11111111'], ex_assign_public_ip=True)