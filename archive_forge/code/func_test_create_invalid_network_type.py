from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_invalid_network_type(self):
    arglist = ['--private', '--project', self._network_segment_range.project_id, '--network-type', 'foo', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])