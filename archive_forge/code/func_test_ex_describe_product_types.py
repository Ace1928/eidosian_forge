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
def test_ex_describe_product_types(self):
    product_types = self.driver.ex_describe_product_types()
    pt = {}
    for e in product_types:
        pt[e['productTypeId']] = e['description']
    self.assertTrue('0001' in pt.keys())
    self.assertTrue('MapR' in pt.values())
    self.assertTrue(pt['0002'] == 'Windows')