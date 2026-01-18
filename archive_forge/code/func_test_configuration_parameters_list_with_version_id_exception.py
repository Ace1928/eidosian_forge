from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
def test_configuration_parameters_list_with_version_id_exception(self):
    args = ['d-123']
    verifylist = [('datastore_version', 'd-123')]
    parsed_args = self.check_parser(self.cmd, args, verifylist)
    self.assertRaises(exceptions.NoUniqueMatch, self.cmd.take_action, parsed_args)