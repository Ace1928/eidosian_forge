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
def test_load_balancer_delete_failure(self):
    arglist = ['unknown_lb']
    verifylist = [('loadbalancer', 'unknown_lb')]
    self.api_mock.load_balancer_list.return_value = {'loadbalancers': []}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.load_balancer_delete)