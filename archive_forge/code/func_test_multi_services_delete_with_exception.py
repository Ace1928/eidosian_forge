from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_multi_services_delete_with_exception(self):
    arglist = [self.services[0].binary, 'unexist_service']
    verifylist = [('service', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    delete_mock_result = [None, exceptions.CommandError]
    self.compute_sdk_client.delete_service = mock.Mock(side_effect=delete_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 compute services failed to delete.', str(e))
    self.compute_sdk_client.delete_service.assert_any_call(self.services[0].binary, ignore_missing=False)
    self.compute_sdk_client.delete_service.assert_any_call('unexist_service', ignore_missing=False)