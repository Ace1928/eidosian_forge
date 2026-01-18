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
def test_baremetal_remove_trait_traits_and_all(self):
    arglist = ['node_uuid', 'CUSTOM_FOO', '--all']
    verifylist = [('node', 'node_uuid'), ('traits', ['CUSTOM_FOO']), ('remove_all', True)]
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
    self.baremetal_mock.node.remove_all_traits.assert_not_called()
    self.baremetal_mock.node.remove_trait.assert_not_called()