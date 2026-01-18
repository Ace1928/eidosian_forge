import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_create_required_args_pass(self):
    """Verifies required arguments."""
    arglist = ['--coe', self.new_ct.coe, '--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id, self.new_ct.name]
    verifylist = [('coe', self.new_ct.coe), ('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id), ('name', self.new_ct.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.cluster_templates_mock.create.assert_called_with(**self.default_create_args)