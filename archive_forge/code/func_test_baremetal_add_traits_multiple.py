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
def test_baremetal_add_traits_multiple(self):
    arglist = ['node_uuid', 'CUSTOM_FOO', 'CUSTOM_BAR']
    verifylist = [('node', 'node_uuid'), ('traits', ['CUSTOM_FOO', 'CUSTOM_BAR'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    expected_calls = [mock.call('node_uuid', 'CUSTOM_FOO'), mock.call('node_uuid', 'CUSTOM_BAR')]
    self.assertEqual(expected_calls, self.baremetal_mock.node.add_trait.call_args_list)