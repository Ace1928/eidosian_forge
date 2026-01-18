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
def test_multi_default_security_group_rules_delete_with_exception(self):
    arglist = [self._default_sg_rules[0].id, 'unexist_rule']
    verifylist = [('rule', [self._default_sg_rules[0].id, 'unexist_rule'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._default_sg_rules[0], exceptions.CommandError]
    self.sdk_client.find_default_security_group_rule = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 default rules failed to delete.', str(e))
    self.sdk_client.find_default_security_group_rule.assert_any_call(self._default_sg_rules[0].id, ignore_missing=False)
    self.sdk_client.find_default_security_group_rule.assert_any_call('unexist_rule', ignore_missing=False)
    self.sdk_client.delete_default_security_group_rule.assert_called_once_with(self._default_sg_rules[0])