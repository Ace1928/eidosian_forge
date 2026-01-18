from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
def test_image_register_required_options(self):
    arglist = ['id', '--username', 'ubuntu']
    verifylist = [('image', 'id'), ('username', 'ubuntu')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_mock.update_image.assert_called_once_with('id', desc=None, user_name='ubuntu')
    expected_columns = ('Description', 'Id', 'Name', 'Status', 'Tags', 'Username')
    self.assertEqual(expected_columns, columns)
    expected_data = ['Image for tests', 'id', 'image', 'Active', '0.1, fake', 'ubuntu']
    self.assertEqual(expected_data, list(data))