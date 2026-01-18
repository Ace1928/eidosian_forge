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
def test_ex_describe_tags(self):
    node = Node('i-4382922a', None, None, None, None, self.driver)
    tags = self.driver.ex_describe_tags(resource=node)
    self.assertEqual(len(tags), 3)
    self.assertTrue('tag' in tags)
    self.assertTrue('owner' in tags)
    self.assertTrue('stack' in tags)