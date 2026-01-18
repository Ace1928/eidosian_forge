from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
def test_image_cache_list(self):
    arglist = []
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.get_image_cache.assert_called()
    self.assertEqual(self.columns, columns)
    self.assertEqual(tuple(self.datalist), tuple(data))