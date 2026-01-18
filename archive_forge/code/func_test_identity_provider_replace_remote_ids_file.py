import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_replace_remote_ids_file(self):
    """Enable Identity Provider.

        Set Identity Provider's ``enabled`` attribute to True.
        """

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        self.new_remote_id = 'new_entity'
        updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
        updated_idp['remote_ids'] = [self.new_remote_id]
        resources = fakes.FakeResource(None, updated_idp, loaded=True)
        self.identity_providers_mock.update.return_value = resources
    prepare(self)
    arglist = ['--enable', identity_fakes.idp_id, '--remote-id-file', self.new_remote_id]
    verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', True), ('disable', False), ('remote_id_file', self.new_remote_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = self.new_remote_id
    with mock.patch('openstackclient.identity.v3.identity_provider.utils.read_blob_file_contents', mocker):
        self.cmd.take_action(parsed_args)
    self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=True, remote_ids=[self.new_remote_id])