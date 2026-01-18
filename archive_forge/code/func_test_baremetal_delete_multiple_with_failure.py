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
def test_baremetal_delete_multiple_with_failure(self):
    arglist = ['xxx-xxxxxx-xxxx', 'badname']
    verifylist = []
    self.baremetal_mock.node.delete.side_effect = ['', exc.ClientException]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
    args = ['xxx-xxxxxx-xxxx', 'badname']
    self.baremetal_mock.node.delete.assert_has_calls([mock.call(x) for x in args])
    self.assertEqual(2, self.baremetal_mock.node.delete.call_count)