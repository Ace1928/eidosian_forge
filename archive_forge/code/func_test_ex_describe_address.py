import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
def test_ex_describe_address(self):
    address = self.driver.ex_describe_address(ex_address_id='01c184f5-1413-4b0b-9f6d-ac993f6c9241')
    self.assertEqual(address['network'], '147.75.33.32')