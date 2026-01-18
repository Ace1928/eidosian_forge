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
@mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
def test_load_actions_with_versioned_args_v39(self, mock_add_arg):
    parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = cinderclient.shell.OpenStackCinderShell()
    shell.subcommands = {}
    shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.9'), False, [])
    self.assertNotIn(mock.call('--foo', help='first foo'), mock_add_arg.call_args_list)
    self.assertIn(mock.call('--foo', help='second foo'), mock_add_arg.call_args_list)