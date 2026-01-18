from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip_association
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_multi_local_ip_associations_delete(self):
    arglist = []
    fixed_port_id = []
    arglist.append(str(self.local_ip))
    for a in self._local_ip_association:
        arglist.append(a.fixed_port_id)
        fixed_port_id.append(a.fixed_port_id)
    verifylist = [('local_ip', str(self.local_ip)), ('fixed_port_id', fixed_port_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._local_ip_association:
        calls.append(call(a.local_ip_id, a.fixed_port_id, ignore_missing=False))
    self.network_client.delete_local_ip_association.assert_has_calls(calls)
    self.assertIsNone(result)