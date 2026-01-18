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
def test_baremetal_set_protected_with_reason(self):
    arglist = ['node_uuid', '--protected', '--protected-reason', 'reason!']
    verifylist = [('nodes', ['node_uuid']), ('protected', True), ('protected_reason', 'reason!')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/protected', 'value': 'True', 'op': 'add'}, {'path': '/protected_reason', 'value': 'reason!', 'op': 'add'}], reset_interfaces=None)