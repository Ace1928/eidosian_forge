import argparse
from unittest import mock
from openstack import exceptions
from openstack.identity.v3 import project
import testtools
from osc_lib.cli import identity as cli_identity
from osc_lib.tests import utils as test_utils
def test_find_project_with_generic_exception(self):
    sdk_connection = mock.Mock()
    sdk_find_project = sdk_connection.identity.find_project
    exc = exceptions.HttpException()
    exc.status_code = 499
    sdk_find_project.side_effect = exc
    with testtools.ExpectedException(exceptions.HttpException):
        cli_identity.find_project(sdk_connection, 'project1')