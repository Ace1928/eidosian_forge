from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_ndp_proxies_delete(self):
    arglist = []
    np_id = []
    for a in self.ndp_proxies:
        arglist.append(a.id)
        np_id.append(a.id)
    verifylist = [('ndp_proxy', np_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_ndp_proxy.assert_has_calls([call(self.ndp_proxy), call(self.ndp_proxy)])
    self.assertIsNone(result)