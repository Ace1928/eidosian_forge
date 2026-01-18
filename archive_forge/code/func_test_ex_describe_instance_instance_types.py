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
def test_ex_describe_instance_instance_types(self):
    instance_types = self.driver.ex_describe_instance_types()
    it = {}
    for e in instance_types:
        it[e['name']] = e['memory']
    self.assertTrue('og4.4xlarge' in it.keys())
    self.assertTrue('oc2.8xlarge' in it.keys())
    self.assertTrue('68718428160' in it.values())
    self.assertTrue(it['m3.large'] == '8050966528')