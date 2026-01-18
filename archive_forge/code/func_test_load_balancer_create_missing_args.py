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
def test_load_balancer_create_missing_args(self, mock_client):
    attrs_list = self.lb_info
    args = ('vip_subnet_id', 'vip_network_id', 'vip_port_id')
    for a in args:
        attrs_list[a] = ''
    for n in range(len(args) + 1):
        for comb in itertools.combinations(args, n):
            filtered_attrs = {k: v for k, v in attrs_list.items() if k not in comb}
            filtered_attrs['wait'] = False
            mock_client.return_value = filtered_attrs
            parsed_args = argparse.Namespace(**filtered_attrs)
            if not any((k in filtered_attrs for k in args)) or all((k in filtered_attrs for k in ('vip_network_id', 'vip_port_id'))):
                self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
            else:
                try:
                    self.cmd.take_action(parsed_args)
                except exceptions.CommandError as e:
                    self.fail('%s raised unexpectedly' % e)