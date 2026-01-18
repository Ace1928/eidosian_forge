import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_manage_baremetal_provision_state_default_wait(self):
    arglist = ['node_uuid', '--wait']
    verifylist = [('nodes', ['node_uuid']), ('provision_state', 'manage'), ('wait_timeout', 0)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    test_node = self.baremetal_mock.node
    test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='manageable', poll_interval=2, timeout=0)