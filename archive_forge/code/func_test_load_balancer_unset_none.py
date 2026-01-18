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
def test_load_balancer_unset_none(self):
    self.api_mock.load_balancer_set.reset_mock()
    arglist = [self._lb.id]
    verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
    verifylist = [('loadbalancer', self._lb.id)] + verifylist
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_set.assert_not_called()