import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
@mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
def test_load_balancer_set_wait(self, mock_attrs, mock_wait):
    qos_policy_id = uuidutils.generate_uuid()
    mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}
    arglist = [self._lb.id, '--name', 'new_name', '--vip-qos-policy-id', qos_policy_id, '--wait']
    verifylist = [('loadbalancer', self._lb.id), ('name', 'new_name'), ('vip_qos_policy_id', qos_policy_id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_set.assert_called_with(self._lb.id, json={'loadbalancer': {'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')