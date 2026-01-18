from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_group_set_filters(self):
    arglist = ['--filters', identity_fakes.endpoint_group_file_path, self.endpoint_group.id]
    verifylist = [('filters', identity_fakes.endpoint_group_file_path), ('endpointgroup', self.endpoint_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = identity_fakes.endpoint_group_filters_2
    with mock.patch('openstackclient.identity.v3.endpoint_group.SetEndpointGroup._read_filters', mocker):
        result = self.cmd.take_action(parsed_args)
    kwargs = {'name': None, 'filters': identity_fakes.endpoint_group_filters_2, 'description': ''}
    self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
    self.assertIsNone(result)