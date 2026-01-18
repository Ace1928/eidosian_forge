import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_trunk_with_subport_invalid_segmentation_id_fail(self):
    subport = self.new_trunk.sub_ports[0]
    arglist = ['--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s,segmentation-type=%(seg_type)s,segmentation-id=boom' % {'seg_type': subport['segmentation_type'], 'port': subport['port_id']}, self.new_trunk.name]
    verifylist = [('name', self.new_trunk.name), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id'], 'segmentation-id': 'boom', 'segmentation-type': subport['segmentation_type']}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual("Segmentation-id 'boom' is not an integer", str(e))