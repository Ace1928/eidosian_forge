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
def test_ex_get_image_from_family(self):
    family = 'coreos-beta'
    description = 'CoreOS beta 522.3.0'
    image = self.driver.ex_get_image_from_family(family)
    self.assertEqual(image.name, 'coreos-beta-522-3-0-v20141226')
    self.assertEqual(image.extra['description'], description)
    self.assertEqual(image.extra['family'], family)
    url = 'https://www.googleapis.com/compute/v1/projects/coreos-cloud/global/images/family/coreos-beta'
    image = self.driver.ex_get_image_from_family(url)
    self.assertEqual(image.name, 'coreos-beta-522-3-0-v20141226')
    self.assertEqual(image.extra['description'], description)
    self.assertEqual(image.extra['family'], family)
    project_list = ['coreos-cloud']
    image = self.driver.ex_get_image_from_family(family, ex_project_list=project_list, ex_standard_projects=False)
    self.assertEqual(image.name, 'coreos-beta-522-3-0-v20141226')
    self.assertEqual(image.extra['description'], description)
    self.assertEqual(image.extra['family'], family)
    self.assertRaises(ResourceNotFoundError, self.driver.ex_get_image_from_family, 'nofamily')