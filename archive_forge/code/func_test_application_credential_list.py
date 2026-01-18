import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_application_credential_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.app_creds_mock.list.assert_called_with(user=None)
    collist = ('ID', 'Name', 'Project ID', 'Description', 'Expires At')
    self.assertEqual(collist, columns)
    datalist = ((identity_fakes.app_cred_id, identity_fakes.app_cred_name, identity_fakes.project_id, None, None),)
    self.assertEqual(datalist, tuple(data))