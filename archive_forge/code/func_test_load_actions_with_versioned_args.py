import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
@mock.patch.object(novaclient.shell.NovaClientArgumentParser, 'add_argument')
def test_load_actions_with_versioned_args(self, mock_add_arg):
    parser = novaclient.shell.NovaClientArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = novaclient.shell.OpenStackComputeShell()
    shell.subcommands = {}
    shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('2.20'), False)
    self.assertIn(mock.call('--foo', help='first foo'), mock_add_arg.call_args_list)
    self.assertNotIn(mock.call('--foo', help='second foo'), mock_add_arg.call_args_list)
    mock_add_arg.reset_mock()
    parser = novaclient.shell.NovaClientArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    shell = novaclient.shell.OpenStackComputeShell()
    shell.subcommands = {}
    shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('2.21'), False)
    self.assertNotIn(mock.call('--foo', help='first foo'), mock_add_arg.call_args_list)
    self.assertIn(mock.call('--foo', help='second foo'), mock_add_arg.call_args_list)