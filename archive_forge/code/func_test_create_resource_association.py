import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_create_resource_association(self):
    fake_bgpvpn = fakes.create_one_bgpvpn()
    fake_res = fakes.create_one_resource()
    fake_res_assoc = fakes.create_one_resource_association(fake_res)
    self.networkclient.create_bgpvpn_router_association = mock.Mock(return_value=fake_res_assoc)
    self.networkclient.find_bgpvpn_fake_resource_association = mock.Mock(side_effect=lambda name_or_id: {'id': name_or_id})
    arglist = [fake_bgpvpn['id'], fake_res['id'], '--project', fake_bgpvpn['tenant_id']]
    verifylist = [('bgpvpn', fake_bgpvpn['id']), ('resource', fake_res['id']), ('project', fake_bgpvpn['tenant_id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cols, data = self.cmd.take_action(parsed_args)
    fake_res_assoc_call = {'fake_resource_id': 'fake_resource_id', 'tenant_id': 'fake_project_id'}
    self.networkclient.create_bgpvpn_router_association.assert_called_once_with(fake_bgpvpn['id'], **fake_res_assoc_call)
    self.assertEqual(sorted_columns, cols)
    self.assertEqual(_get_data(fake_res_assoc), data)