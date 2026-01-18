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
def test_baremetal_unset_target_raid_config(self):
    self.cmd.log = mock.Mock(autospec=True)
    arglist = ['node_uuid', '--target-raid-config']
    verifylist = [('nodes', ['node_uuid']), ('target_raid_config', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.cmd.log.warning.assert_not_called()
    self.assertFalse(self.baremetal_mock.node.update.called)
    self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', {})