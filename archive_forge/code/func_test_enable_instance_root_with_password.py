from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient.osc.v1 import database_root
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_enable_instance_root_with_password(self, mock_find):
    args = ['1234', '--root_password', 'secret']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.root_client.create_instance_root(None, root_password='secret')