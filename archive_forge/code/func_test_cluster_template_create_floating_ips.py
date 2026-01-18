import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_create_floating_ips(self):
    """Verifies floating ip parameters."""
    arglist = ['--coe', self.new_ct.coe, '--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id, '--floating-ip-enabled', self.new_ct.name]
    verifylist = [('coe', self.new_ct.coe), ('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id), ('floating_ip_enabled', [True]), ('name', self.new_ct.name)]
    self.default_create_args['floating_ip_enabled'] = True
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.default_create_args.pop('floating_ip_enabled')
    arglist.append('--floating-ip-disabled')
    verifylist.remove(('floating_ip_enabled', [True]))
    verifylist.append(('floating_ip_enabled', [True, False]))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(InvalidAttribute, self.cmd.take_action, parsed_args)