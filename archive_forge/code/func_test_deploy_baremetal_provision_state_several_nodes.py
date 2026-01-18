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
def test_deploy_baremetal_provision_state_several_nodes(self):
    arglist = ['node_uuid', 'node_name', '--wait', '15']
    verifylist = [('nodes', ['node_uuid', 'node_name']), ('provision_state', 'active'), ('wait_timeout', 15)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    test_node = self.baremetal_mock.node
    test_node.set_provision_state.assert_has_calls([mock.call(n, 'active', cleansteps=None, deploysteps=None, configdrive=None, rescue_password=None) for n in ['node_uuid', 'node_name']])
    test_node.wait_for_provision_state.assert_called_once_with(['node_uuid', 'node_name'], expected_state='active', poll_interval=10, timeout=15)