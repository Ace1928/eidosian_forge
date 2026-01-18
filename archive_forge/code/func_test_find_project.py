import argparse
from unittest import mock
from openstack import exceptions
from openstack.identity.v3 import project
import testtools
from osc_lib.cli import identity as cli_identity
from osc_lib.tests import utils as test_utils
def test_find_project(self):
    sdk_connection = mock.Mock()
    sdk_find_project = sdk_connection.identity.find_project
    sdk_find_project.return_value = mock.sentinel.project1
    ret = cli_identity.find_project(sdk_connection, 'project1')
    self.assertEqual(mock.sentinel.project1, ret)
    sdk_find_project.assert_called_once_with('project1', ignore_missing=False, domain_id=None)