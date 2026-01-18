from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
@ddt.data((True, True), (False, False), (None, 'N/A'))
@ddt.unpack
def test_is_public(self, is_public, expected):
    fake_share_group_type_info = {'name': 'fake_name'}
    if is_public is not None:
        fake_share_group_type_info['is_public'] = is_public
    share_group_type = types.ShareGroupType(self.manager, fake_share_group_type_info, loaded=True)
    result = share_group_type.is_public
    self.assertEqual(expected, result)