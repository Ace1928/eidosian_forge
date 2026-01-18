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
def ex_register_image(self):
    mapping = [{'DeviceName': '/dev/sda1', 'Ebs': {'SnapshotId': 'snap-5ade3e4e'}}]
    image = self.driver.ex_register_image(name='Test Image', root_device_name='/dev/sda1', description='My Image', architecture='x86_64', block_device_mapping=mapping, ena_support=True, billing_products=['ab-5dh78019'], sriov_net_support='simple')
    self.assertEqual(image.id, 'ami-57c2fb3e')