from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_invalid_version_strings(self):
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '200')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.1.4')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '200.23.66.3')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5 .3')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5. 3')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '5.03')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '02.1')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.001')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, ' 2.1')
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.APIVersion, '2.1 ')