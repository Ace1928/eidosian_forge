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
def test_load_balancer_remove_qos_policy(self, mock_attrs):
    mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'vip_qos_policy_id': None}
    arglist = [self._lb.id, '--vip-qos-policy-id', 'None']
    verifylist = [('loadbalancer', self._lb.id), ('vip_qos_policy_id', 'None')]
    try:
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
    except Exception as e:
        self.fail('%s raised unexpectedly' % e)