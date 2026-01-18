from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_port_forwardings_delete(self):
    arglist = []
    pf_id = []
    arglist.append(str(self.floating_ip))
    for a in self._port_forwarding:
        arglist.append(a.id)
        pf_id.append(a.id)
    verifylist = [('floating_ip', str(self.floating_ip)), ('port_forwarding_id', pf_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._port_forwarding:
        calls.append(call(a.floatingip_id, a.id, ignore_missing=False))
    self.network_client.delete_floating_ip_port_forwarding.assert_has_calls(calls)
    self.assertIsNone(result)