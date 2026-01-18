import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7policy_list_with_not_any_tags(self):
    arglist = ['--not-any-tags', 'foo,bar']
    verifylist = [('not_any_tags', ['foo', 'bar'])]
    expected_attrs = {'not-tags-any': ['foo', 'bar']}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_list.assert_called_with(**expected_attrs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))