import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_list_with_target_and_resource(self):
    arglist = []
    verifylist = []
    target_id = 'aaaaaaaa-aaaa-aaaa-aaaaaaaaaaaaaaaaa'
    resource_id = 'bbbbbbbb-bbbb-bbbb-bbbbbbbbbbbbbbbbb'
    log = fakes.NetworkLog().create({'target_id': target_id, 'resource_id': resource_id})
    self.mocked.return_value = {'logs': [log]}
    logged = 'Logged: (security_group) %(res_id)s on (port) %(t_id)s' % {'res_id': resource_id, 't_id': target_id}
    expect_log = copy.deepcopy(log)
    expect_log.update({'resource': logged, 'event': 'Event: ALL'})
    self._setup_summary(expect=expect_log)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    headers, data = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with()
    self.assertEqual(list(self.short_header), headers)
    self.assertEqual([self.short_data], list(data))