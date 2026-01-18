from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_metadef_property_create_invalid_schema(self):
    arglist = ['--name', 'cpu_cores', '--schema', '{invalid}', '--title', 'vCPU Cores', '--type', 'integer', self._metadef_namespace.namespace]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)