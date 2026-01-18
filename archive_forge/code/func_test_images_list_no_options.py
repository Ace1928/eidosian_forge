from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import images as api_images
from saharaclient.osc.v1 import images as osc_images
from saharaclient.tests.unit.osc.v1 import test_images as images_v1
def test_images_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Username', 'Tags']
    self.assertEqual(expected_columns, columns)
    expected_data = [('image', 'id', 'ubuntu', '0.1, fake')]
    self.assertEqual(expected_data, list(data))