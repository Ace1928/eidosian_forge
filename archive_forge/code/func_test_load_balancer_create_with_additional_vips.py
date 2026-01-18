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
@mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
def test_load_balancer_create_with_additional_vips(self, mock_client):
    mock_client.return_value = self.lb_info
    arglist = ['--name', self._lb.name, '--vip-subnet-id', self._lb.vip_subnet_id, '--project', self._lb.project_id, '--additional-vip', 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[0]['subnet_id'], self._lb.additional_vips[0]['ip_address']), '--additional-vip', 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[1]['subnet_id'], self._lb.additional_vips[1]['ip_address'])]
    verifylist = [('name', self._lb.name), ('vip_subnet_id', self._lb.vip_subnet_id), ('project', self._lb.project_id), ('additional_vip', ['subnet-id={},ip-address={}'.format(self._lb.additional_vips[0]['subnet_id'], self._lb.additional_vips[0]['ip_address']), 'subnet-id={},ip-address={}'.format(self._lb.additional_vips[1]['subnet_id'], self._lb.additional_vips[1]['ip_address'])])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': self.lb_info})