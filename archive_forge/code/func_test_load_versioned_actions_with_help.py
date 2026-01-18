import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_load_versioned_actions_with_help(self):
    parser = cinderclient.shell.CinderClientArgumentParser()
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = cinderclient.shell.OpenStackCinderShell()
    shell.subcommands = {}
    with mock.patch.object(subparsers, 'add_parser') as mock_add_parser:
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.1'), True, [])
        self.assertIn('fake-action', shell.subcommands.keys())
        expected_help = 'help message (Supported by API versions %(start)s - %(end)s)' % {'start': '3.0', 'end': '3.3'}
        expected_desc = 'help message\n\n    This will not show up in help message\n    '
        mock_add_parser.assert_any_call('fake-action', help=expected_help, description=expected_desc, add_help=False, formatter_class=cinderclient.shell.OpenStackHelpFormatter)