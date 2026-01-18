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
def test_load_balancer_create_with_provider(self, mock_client):
    provider = 'foobar'
    lb_info = copy.deepcopy(self.lb_info)
    lb_info.update({'provider': provider})
    mock_client.return_value = lb_info
    arglist = ['--name', self._lb.name, '--vip-network-id', self._lb.vip_network_id, '--project', self._lb.project_id, '--provider', provider]
    verifylist = [('name', self._lb.name), ('vip_network_id', self._lb.vip_network_id), ('project', self._lb.project_id), ('provider', provider)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_create.assert_called_with(json={'loadbalancer': lb_info})