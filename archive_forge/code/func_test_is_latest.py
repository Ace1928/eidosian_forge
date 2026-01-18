from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_is_latest(self):
    v1 = api_versions.APIVersion('1.0')
    self.assertFalse(v1.is_latest())
    v_latest = api_versions.APIVersion(api_versions.MAX_VERSION)
    self.assertTrue(v_latest.is_latest())