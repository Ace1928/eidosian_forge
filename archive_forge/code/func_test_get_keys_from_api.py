import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
@ddt.data({'prefer_resource_data': True, 'resource_extra_specs': {}}, {'prefer_resource_data': False, 'resource_extra_specs': {'fake': 'fake'}}, {'prefer_resource_data': False, 'resource_extra_specs': {}})
@ddt.unpack
def test_get_keys_from_api(self, prefer_resource_data, resource_extra_specs):
    manager = mock.Mock()
    valid_extra_specs = {'test': 'test'}
    manager.api.client.get = mock.Mock(return_value=(200, {'extra_specs': valid_extra_specs}))
    info = {'name': 'test', 'uuid': 'fake', 'extra_specs': resource_extra_specs}
    share_type = share_types.ShareType(manager, info, loaded=True)
    actual_result = share_type.get_keys(prefer_resource_data)
    self.assertEqual(actual_result, valid_extra_specs)
    self.assertEqual(manager.api.client.get.call_count, 1)