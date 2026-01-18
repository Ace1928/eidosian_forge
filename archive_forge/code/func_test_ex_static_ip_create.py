import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_ex_static_ip_create(self):
    result = self.driver.ex_static_ip_create()
    self.assertEqual(len(result), 2)
    self.assertEqual(len(list(result[0].keys())), 6)
    self.assertEqual(len(list(result[1].keys())), 6)