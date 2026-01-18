from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_metadef_property_delete_missing_arguments(self):
    arglist = []
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])
    arglist = ['namespace']
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])