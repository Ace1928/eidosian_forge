import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import amphora
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_amphora_list_no_options(self):
    arglist = []
    verify_list = []
    parsed_args = self.check_parser(self.cmd, arglist, verify_list)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.amphora_list.assert_called_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data_list, tuple(data))