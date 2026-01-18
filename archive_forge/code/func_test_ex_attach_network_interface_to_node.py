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
def test_ex_attach_network_interface_to_node(self):
    node = self.driver.list_nodes()[0]
    interface = self.driver.ex_list_network_interfaces()[0]
    try:
        self.driver.ex_attach_network_interface_to_node(interface, node, 1)
    except NotImplementedError:
        pass
    else:
        self.fail('Exception was not thrown')