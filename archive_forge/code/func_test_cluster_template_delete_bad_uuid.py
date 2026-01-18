import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_delete_bad_uuid(self):
    self.cluster_templates_mock.delete.side_effect = osc_exceptions.NotFound(404)
    arglist = ['foo']
    verifylist = [('cluster-templates', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    returns = self.cmd.take_action(parsed_args)
    self.assertEqual(returns, None)