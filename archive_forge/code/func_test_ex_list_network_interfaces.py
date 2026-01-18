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
def test_ex_list_network_interfaces(self):
    interfaces = self.driver.ex_list_network_interfaces()
    self.assertEqual(len(interfaces), 2)
    self.assertEqual('eni-18e6c05e', interfaces[0].id)
    self.assertEqual('in-use', interfaces[0].state)
    self.assertEqual('0e:6e:df:72:78:af', interfaces[0].extra['mac_address'])
    self.assertEqual('eni-83e3c5c5', interfaces[1].id)
    self.assertEqual('in-use', interfaces[1].state)
    self.assertEqual('0e:93:0b:e9:e9:c4', interfaces[1].extra['mac_address'])