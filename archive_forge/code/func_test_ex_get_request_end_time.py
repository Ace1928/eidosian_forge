import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.test.secrets import VCL_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcl import VCLNodeDriver as VCL
def test_ex_get_request_end_time(self):
    node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
    self.assertEqual(self.driver.ex_get_request_end_time(node), 1334168100)