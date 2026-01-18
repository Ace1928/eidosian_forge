from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
def test_images_list_successful_selection(self):
    arglist = ['--name', 'image', '--tags', 'fake', '--username', 'ubuntu']
    verifylist = [('name', 'image'), ('tags', ['fake']), ('username', 'ubuntu')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_mock.list.assert_called_once_with(search_opts={'tags': ['fake']})
    expected_columns = ['Name', 'Id', 'Username', 'Tags']
    self.assertEqual(expected_columns, columns)
    expected_data = [('image', 'id', 'ubuntu', '0.1, fake')]
    self.assertEqual(expected_data, list(data))