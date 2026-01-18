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
def test_ex_create_node_with_ex_blockdevicemappings(self):
    EC2MockHttp.type = 'create_ex_blockdevicemappings'
    image = NodeImage(id='ami-be3adfd7', name=self.image_name, driver=self.driver)
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    mappings = [{'DeviceName': '/dev/sda1', 'Ebs.VolumeSize': 10}, {'DeviceName': '/dev/sdb', 'VirtualName': 'ephemeral0'}, {'DeviceName': '/dev/sdc', 'VirtualName': 'ephemeral1'}]
    node = self.driver.create_node(name='foo', image=image, size=size, ex_blockdevicemappings=mappings)
    self.assertEqual(node.id, 'i-2ba64342')