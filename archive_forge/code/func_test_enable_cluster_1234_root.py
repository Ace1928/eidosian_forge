from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient.osc.v1 import database_root
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_enable_cluster_1234_root(self, mock_find):
    mock_find.side_effect = [exceptions.CommandError(), (None, 'cluster')]
    self.root_client.create_cluster_root.return_value = self.data['cluster']
    args = ['1234']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(('root', 'password'), data)