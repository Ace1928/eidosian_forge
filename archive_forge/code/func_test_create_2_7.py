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
@ddt.data(*get_valid_type_create_data_2_0())
@ddt.unpack
def test_create_2_7(self, is_public, dhss, snapshot, extra_specs):
    extra_specs = copy.copy(extra_specs)
    manager = self._get_share_types_manager('2.7')
    self.mock_object(manager, '_create', mock.Mock(return_value='fake'))
    result = manager.create('test-type-3', spec_driver_handles_share_servers=dhss, spec_snapshot_support=snapshot, extra_specs=extra_specs, is_public=is_public)
    if extra_specs is None:
        extra_specs = {}
    expected_extra_specs = dict(extra_specs)
    expected_body = {'share_type': {'name': 'test-type-3', 'share_type_access:is_public': is_public, 'extra_specs': expected_extra_specs}}
    expected_body['share_type']['extra_specs']['driver_handles_share_servers'] = dhss
    expected_body['share_type']['extra_specs']['snapshot_support'] = True if snapshot is None else snapshot
    manager._create.assert_called_once_with('/types', expected_body, 'share_type')
    self.assertEqual('fake', result)