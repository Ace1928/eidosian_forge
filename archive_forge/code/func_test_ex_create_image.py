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
def test_ex_create_image(self):
    volume = self.driver.ex_get_volume('lcdisk')
    description = 'CoreOS, CoreOS stable, 1520.6.0, amd64-usr published on 2017-10-12'
    name = 'coreos'
    family = 'coreos-stable'
    licenses = ['projects/coreos-cloud/global/licenses/coreos-stable']
    guest_os_features = ['VIRTIO_SCSI_MULTIQUEUE']
    expected_features = [{'type': 'VIRTIO_SCSI_MULTIQUEUE'}]
    mock_request = mock.Mock()
    mock_request.side_effect = self.driver.connection.async_request
    self.driver.connection.async_request = mock_request
    image = self.driver.ex_create_image(name, volume, description=description, family=family, guest_os_features=guest_os_features, ex_licenses=licenses)
    self.assertTrue(isinstance(image, GCENodeImage))
    self.assertTrue(image.name.startswith(name))
    self.assertEqual(image.extra['description'], description)
    self.assertEqual(image.extra['family'], family)
    self.assertEqual(image.extra['guestOsFeatures'], expected_features)
    self.assertEqual(image.extra['licenses'][0].name, licenses[0].split('/')[-1])
    expected_data = {'description': description, 'family': family, 'guestOsFeatures': expected_features, 'name': name, 'licenses': licenses, 'sourceDisk': volume.extra['selfLink'], 'zone': volume.extra['zone'].name}
    mock_request.assert_called_once_with('/global/images', data=expected_data, method='POST')