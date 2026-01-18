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
def test_load_balancer_create_wait(self, mock_client, mock_wait):
    mock_client.return_value = self.lb_info
    self.api_mock.load_balancer_show.return_value = self.lb_info
    arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--flavor', self._lb.flavor_id, '--wait']
    verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('flavor', self._lb.flavor_id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': self.lb_info})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.lb_info['id'], sleep_time=mock.ANY, status_field='provisioning_status')