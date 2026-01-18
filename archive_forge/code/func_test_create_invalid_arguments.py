from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_groups
def test_create_invalid_arguments(self):
    create_args = {'name': fake.ShareGroup.name, 'description': fake.ShareGroup.description, 'share_types': [fake.ShareType().id], 'source_share_group_snapshot': fake.ShareGroupSnapshot()}
    self.assertRaises(ValueError, self.manager.create, **create_args)