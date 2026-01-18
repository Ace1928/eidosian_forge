from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_group_set_description(self):
    arglist = ['--description', 'qwerty', self.endpoint_group.id]
    verifylist = [('description', 'qwerty'), ('endpointgroup', self.endpoint_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': None, 'filters': None, 'description': 'qwerty'}
    self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
    self.assertIsNone(result)