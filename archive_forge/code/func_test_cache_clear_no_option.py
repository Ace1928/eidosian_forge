from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
def test_cache_clear_no_option(self):
    arglist = []
    verifylist = [('target', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.assertIsNone(self.image_client.clear_cache.assert_called_with(None))