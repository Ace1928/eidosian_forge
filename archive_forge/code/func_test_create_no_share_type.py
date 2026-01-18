from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
def test_create_no_share_type(self):
    create_args = {'name': fake.ShareGroupType.name, 'share_types': [], 'is_public': False, 'group_specs': self.fake_group_specs}
    self.assertRaises(ValueError, self.manager.create, **create_args)