from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
def test_image_tags_set(self):
    arglist = ['image', '--tags', 'fake', '0.1']
    verifylist = [('image', 'image'), ('tags', ['fake', '0.1'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_mock.find_unique.assert_called_with(name='image')
    self.image_mock.update_tags.assert_called_once_with('id', ['fake', '0.1'])