import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.data(*unit_test_types.get_valid_type_create_data_2_24())
@ddt.unpack
def test_create_delete_share_type_2_24(self, is_public, dhss, spec_snapshot_support, spec_create_share_from_snapshot, extra_specs):
    self.skip_if_microversion_not_supported('2.24')
    self._test_create_delete_share_type('2.24', is_public, dhss, spec_snapshot_support, spec_create_share_from_snapshot, None, None, extra_specs)