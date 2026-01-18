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
def test_baremetal_list_traits(self):
    arglist = ['node_uuid']
    verifylist = [('node', 'node_uuid')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.get_traits.assert_called_once_with('node_uuid')
    self.assertEqual(('Traits',), columns)
    self.assertEqual([[trait] for trait in baremetal_fakes.TRAITS], data)