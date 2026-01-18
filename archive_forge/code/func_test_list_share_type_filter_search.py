import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def test_list_share_type_filter_search(self):
    extra_specs = {'aaaa': 'bbbb'}
    name1 = data_utils.rand_name('manilaclient_functional_test1')
    self.create_share_type(name=name1, driver_handles_share_servers='False')
    name2 = data_utils.rand_name('manilaclient_functional_test2')
    self.create_share_type(name=name2, extra_specs=extra_specs, driver_handles_share_servers='True')
    list_all = False
    search_opts = {'extra_specs': extra_specs}
    share_types = self.admin_client.list_share_types(list_all=list_all, search_opts=search_opts, microversion='2.43')
    self.assertTrue(share_types is not None)
    expect = 'aaaa : bbbb'
    self.assertTrue(len(share_types) == 1)
    self.assertTrue(all(('optional_extra_specs' in s for s in share_types)))
    self.assertTrue(all((s['Name'] == name2 for s in share_types)))
    self.assertTrue(all((s['optional_extra_specs'] == expect for s in share_types)))