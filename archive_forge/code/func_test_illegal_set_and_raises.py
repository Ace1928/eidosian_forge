import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
def test_illegal_set_and_raises(self):
    self.neutronclient.update_network_log = mock.Mock(side_effect=Exception)
    target = self.res['id']
    arglist = [target, '--name', 'my-name']
    verifylist = [('network_log', target), ('name', 'my-name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)