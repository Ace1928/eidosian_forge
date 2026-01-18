from unittest import mock
from unittest.mock import call
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import (
from openstack.test import fakes as sdk_fakes
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import default_security_group_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_default_security_group_rules_delete(self):
    arglist = []
    verifylist = []
    for s in self._default_sg_rules:
        arglist.append(s.id)
    verifylist = [('rule', arglist)]
    self.sdk_client.find_default_security_group_rule.side_effect = self._default_sg_rules
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self._default_sg_rules:
        calls.append(call(s))
    self.sdk_client.delete_default_security_group_rule.assert_has_calls(calls)
    self.assertIsNone(result)