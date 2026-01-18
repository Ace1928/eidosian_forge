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
def test_ex_describe_all_addresses(self):
    EC2MockHttp.type = 'all_addresses'
    elastic_ips1 = self.driver.ex_describe_all_addresses()
    elastic_ips2 = self.driver.ex_describe_all_addresses(only_associated=True)
    self.assertEqual('1.2.3.7', elastic_ips1[3].ip)
    self.assertEqual('vpc', elastic_ips1[3].domain)
    self.assertEqual('eipalloc-992a5cf8', elastic_ips1[3].extra['allocation_id'])
    self.assertEqual(len(elastic_ips2), 2)
    self.assertEqual('1.2.3.5', elastic_ips2[1].ip)
    self.assertEqual('vpc', elastic_ips2[1].domain)