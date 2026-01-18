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
def test_ex_copy_image(self):
    name = 'coreos'
    url = 'gs://storage.core-os.net/coreos/amd64-generic/247.0.0/coreos_production_gce.tar.gz'
    description = 'CoreOS, CoreOS stable, 1520.6.0, amd64-usr published on 2017-10-12'
    family = 'coreos-stable'
    guest_os_features = ['VIRTIO_SCSI_MULTIQUEUE']
    expected_features = [{'type': 'VIRTIO_SCSI_MULTIQUEUE'}]
    image = self.driver.ex_copy_image(name, url, description=description, family=family, guest_os_features=guest_os_features)
    self.assertTrue(image.name.startswith(name))
    self.assertEqual(image.extra['description'], description)
    self.assertEqual(image.extra['family'], family)
    self.assertEqual(image.extra['guestOsFeatures'], expected_features)