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
def test_authorize_security_group_ingress(self):
    ranges = ['1.1.1.1/32', '2.2.2.2/32']
    description = 'automated authorised IP ingress test'
    resp = self.driver.ex_authorize_security_group_ingress('sg-42916629', 22, 22, cidr_ips=ranges, description=description)
    self.assertTrue(resp)
    groups = [{'group_id': 'sg-949265ff'}]
    description = 'automated authorised group ingress test'
    resp = self.driver.ex_authorize_security_group_ingress('sg-42916629', 22, 23, group_pairs=groups, description=description)
    self.assertTrue(resp)