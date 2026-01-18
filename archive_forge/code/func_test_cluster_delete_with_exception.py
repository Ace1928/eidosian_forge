from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_cluster_delete_with_exception(self, mock_find):
    args = ['fakecluster']
    parsed_args = self.check_parser(self.cmd, args, [])
    mock_find.side_effect = exceptions.CommandError
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)