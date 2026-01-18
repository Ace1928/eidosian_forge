from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import node_group_templates as osc_ngt
from saharaclient.tests.unit.osc.v1 import fakes
def test_ngt_update_private_unprotected(self):
    arglist = ['template', '--private', '--unprotected']
    verifylist = [('node_group_template', 'template'), ('is_public', False), ('is_protected', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ngt_mock.update.assert_called_once_with('ng_id', is_protected=False, is_public=False)