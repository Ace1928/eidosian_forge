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
def test_ex_get_image(self):
    partial_name = 'debian-7'
    image = self.driver.ex_get_image(partial_name)
    self.assertEqual(image.name, 'debian-7-wheezy-v20131120')
    self.assertTrue(image.extra['description'].startswith('Debian'))
    partial_name = 'debian-6'
    image = self.driver.ex_get_image(partial_name)
    self.assertEqual(image.name, 'debian-6-squeeze-v20130926')
    self.assertTrue(image.extra['description'].startswith('Debian'))
    partial_name = 'debian-7'
    image = self.driver.ex_get_image(partial_name, ['debian-cloud'])
    self.assertEqual(image.name, 'debian-7-wheezy-v20131120')
    partial_name = 'debian-7'
    self.assertRaises(ResourceNotFoundError, self.driver.ex_get_image, partial_name, 'suse-cloud', ex_standard_projects=False)