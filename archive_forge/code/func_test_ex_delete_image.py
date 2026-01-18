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
def test_ex_delete_image(self):
    self.assertRaises(ResourceNotFoundError, self.driver.ex_get_image, 'missing-image')
    self.assertRaises(ResourceNotFoundError, self.driver.ex_delete_image, 'missing-image')
    image = self.driver.ex_get_image('debian-7')
    deleted = self.driver.ex_delete_image(image)
    self.assertTrue(deleted)