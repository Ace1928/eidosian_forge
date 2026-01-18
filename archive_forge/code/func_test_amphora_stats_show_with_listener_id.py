import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import amphora
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
def test_amphora_stats_show_with_listener_id(self, mock_get_listener_attrs):
    listener_id = list(self.stats)[0]
    arglist = ['--listener', listener_id, self._amp.id]
    verifylist = [('amphora_id', self._amp.id)]
    mock_get_listener_attrs.return_value = {'listener_id': listener_id}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.amphora_stats_show.assert_called_with(amphora_id=self._amp.id)
    column_idx = columns.index('bytes_in')
    bytes_in = self.stats[listener_id]
    self.assertEqual(data[column_idx], bytes_in)