import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_get_subnetwork(self):
    name = 'cf-972cf02e6ad49112'
    region_name = 'us-central1'
    region = self.driver.ex_get_region(region_name)
    subnetwork = self.driver.ex_get_subnetwork(name)
    self.assertEqual(subnetwork.name, name)
    subnetwork = self.driver.ex_get_subnetwork(name, region_name)
    self.assertEqual(subnetwork.name, name)
    subnetwork = self.driver.ex_get_subnetwork(name, region)
    self.assertEqual(subnetwork.name, name)
    url = 'https://www.googleapis.com/compute/v1/projects/project_name/regions/us-central1/subnetworks/cf-972cf02e6ad49112'
    subnetwork = self.driver.ex_get_subnetwork(url)
    self.assertEqual(subnetwork.name, name)
    self.assertEqual(subnetwork.region.name, region_name)
    url_other = 'https://www.googleapis.com/compute/v1/projects/other_name/regions/us-central1/subnetworks/cf-972cf02e6ad49114'
    subnetwork = self.driver.ex_get_subnetwork(url_other)
    self.assertEqual(subnetwork.name, 'cf-972cf02e6ad49114')