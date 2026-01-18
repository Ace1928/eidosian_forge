from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_api_version_is_null(self):
    headers = {}
    api_versions.update_headers(headers, api_versions.APIVersion())
    self.assertEqual({}, headers)