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
def test_ex_change_node_size_same_size(self):
    size = NodeSize('m1.small', 'Small Instance', None, None, None, None, driver=self.driver)
    node = Node('i-4382922a', None, None, None, None, self.driver, extra={'instancetype': 'm1.small'})
    try:
        self.driver.ex_change_node_size(node=node, new_size=size)
    except ValueError:
        pass
    else:
        self.fail('Same size was passed, but an exception was not thrown')