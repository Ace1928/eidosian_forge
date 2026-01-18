from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_type_access as type_access
def test_list_using_unsupported_microversion(self):
    fake_share_group_type_access = fake.ShareGroupTypeAccess()
    self.manager.api.api_version = manilaclient.API_MIN_VERSION
    self.assertRaises(exceptions.UnsupportedVersion, self.manager.list, fake_share_group_type_access)