import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
def test_common_args_in_help_message(self):
    expected_args = ('--version', '', '--debug', '--os-cache', '--os-reset-cache', '--os-user-id', '--os-username', '--os-password', '--os-tenant-name', '--os-project-name', '--os-tenant-id', '--os-project-id', '--os-user-domain-id', '--os-user-domain-name', '--os-project-domain-id', '--os-project-domain-name', '--os-auth-url', '--os-region-name', '--service-type', '--service-name', '--share-service-name', '--endpoint-type', '--os-share-api-version', '--os-cacert', '--retries', '--os-cert', '--os-key')
    help_text = self.shell('help')
    for expected_arg in expected_args:
        self.assertIn(expected_arg, help_text)