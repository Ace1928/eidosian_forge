from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import cache
from openstackclient.tests.unit.image.v2 import fakes
def test_cache_queue_multiple_images(self):
    images = fakes.create_images(count=3)
    arglist = [i.id for i in images]
    verifylist = [('images', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.image_client.find_image.side_effect = images
    self.cmd.take_action(parsed_args)
    calls = [call(i.id) for i in images]
    self.image_client.queue_image.assert_has_calls(calls)