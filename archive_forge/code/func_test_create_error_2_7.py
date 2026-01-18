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
@ddt.data((False, False, True, {'snapshot_support': True, 'replication_type': 'fake_repl_type'}), (False, False, False, {'snapshot_support': False, 'replication_type': 'fake_repl_type'}), (False, False, True, {'snapshot_support': False, 'replication_type': 'fake_repl_type'}), (False, False, False, {'snapshot_support': True, 'replication_type': 'fake_repl_type'}), (False, True, None, {'driver_handles_share_servers': True}), (False, False, None, {'driver_handles_share_servers': True}), (False, None, None, {'driver_handles_share_servers': True}), (False, None, None, {'driver_handles_share_servers': None}))
@ddt.unpack
def test_create_error_2_7(self, is_public, dhss, snapshot, extra_specs):
    manager = self._get_share_types_manager('2.7')
    self.mock_object(manager, '_create', mock.Mock(return_value='fake'))
    self.assertRaises(exceptions.CommandError, manager.create, 'test-type-3', spec_driver_handles_share_servers=dhss, spec_snapshot_support=snapshot, extra_specs=extra_specs, is_public=is_public)