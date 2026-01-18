import operator
from unittest import mock
from osc_lib.tests.utils import ParserException
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def test_create_router_association(self):
    fake_res_assoc = fakes.create_one_resource_association(self.fake_router)
    self.networkclient.create_bgpvpn_router_association = mock.Mock(return_value={fakes.BgpvpnFakeRouterAssoc._resource: fake_res_assoc, 'advertise_extra_routes': True})
    arglist = self._build_args()
    verifylist = self._build_verify_list(('advertise_extra_routes', False))
    self._exec_create_router_association(fake_res_assoc, arglist, verifylist)