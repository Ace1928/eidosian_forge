import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import amphora
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_amphora_attrs')
def test_amphora_show(self, mock_client):
    mock_client.return_value = {'amphora_id': self._amp.id}
    arglist = [self._amp.id]
    verify_list = [('amphora_id', self._amp.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verify_list)
    rows, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.rows, rows)
    self.api_mock.amphora_show.assert_called_with(amphora_id=self._amp.id)