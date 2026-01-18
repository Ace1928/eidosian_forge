import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_create_router_association_no_advertise(self):
    fake_res_assoc = fakes.create_one_resource_association(self.fake_router, {'advertise_extra_routes': False})
    self.networkclient.create_bgpvpn_router_association = mock.Mock(return_value=fake_res_assoc)
    arglist = self._build_args('--no-advertise_extra_routes')
    verifylist = self._build_verify_list(('advertise_extra_routes', False))
    cols, data = self._exec_create_router_association(fake_res_assoc, arglist, verifylist)
    self.assertEqual(sorted_columns, cols)
    self.assertEqual(_get_data(fake_res_assoc), data)