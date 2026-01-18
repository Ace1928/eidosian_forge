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
def test_create_volume_image_family(self):
    volume_name = 'lcdisk'
    size = 10
    volume = self.driver.create_volume(size, volume_name, ex_image_family='coreos-stable')
    self.assertTrue(isinstance(volume, StorageVolume))
    self.assertEqual(volume.name, volume_name)
    image = self.driver.ex_get_image('debian-7')
    self.assertRaises(ValueError, self.driver.create_volume, size, volume_name, image=image, ex_image_family='coreos-stable')