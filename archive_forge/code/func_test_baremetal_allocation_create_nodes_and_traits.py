import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_create_nodes_and_traits(self):
    arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--candidate-node', 'node1', '--trait', 'CUSTOM_1', '--candidate-node', 'node2', '--trait', 'CUSTOM_2']
    verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('candidate_nodes', ['node1', 'node2']), ('traits', ['CUSTOM_1', 'CUSTOM_2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    args = {'resource_class': baremetal_fakes.baremetal_resource_class, 'candidate_nodes': ['node1', 'node2'], 'traits': ['CUSTOM_1', 'CUSTOM_2']}
    self.baremetal_mock.allocation.create.assert_called_once_with(**args)