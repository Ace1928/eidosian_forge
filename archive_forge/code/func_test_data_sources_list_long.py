from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import data_sources as api_ds
from saharaclient.osc.v1 import data_sources as osc_ds
from saharaclient.tests.unit.osc.v1 import test_data_sources as tds_v1
def test_data_sources_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Id', 'Type', 'Url', 'Description', 'Is public', 'Is protected']
    self.assertEqual(expected_columns, columns)
    expected_data = [('source', 'id', 'swift', 'swift://container.sahara/object', 'Data Source for tests', True, True)]
    self.assertEqual(expected_data, list(data))