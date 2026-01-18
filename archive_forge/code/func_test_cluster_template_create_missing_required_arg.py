import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_create_missing_required_arg(self):
    """Verifies missing required arguments."""
    arglist = ['--external-network', self.new_ct.external_network_id, '--image', self.new_ct.image_id]
    verifylist = [('external_network', self.new_ct.external_network_id), ('image', self.new_ct.image_id)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)
    arglist.append('--coe')
    arglist.append(self.new_ct.coe)
    verifylist.append(('coe', self.new_ct.coe))
    arglist.remove('--image')
    arglist.remove(self.new_ct.image_id)
    verifylist.remove(('image', self.new_ct.image_id))
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)
    arglist.remove('--external-network')
    arglist.remove(self.new_ct.external_network_id)
    verifylist.remove(('external_network', self.new_ct.external_network_id))
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)