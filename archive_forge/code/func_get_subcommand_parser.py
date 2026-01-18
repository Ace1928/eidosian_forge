import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def get_subcommand_parser(self, version):
    parser = self.get_base_parser()
    self.subcommands = {}
    subparsers = parser.add_subparsers(metavar='<subcommand>')
    try:
        actions_modules = {'1': shell_v1.COMMAND_MODULES}[version]
    except KeyError:
        actions_modules = shell_v1.COMMAND_MODULES
    for actions_module in actions_modules:
        self._find_actions(subparsers, actions_module)
    self._find_actions(subparsers, self)
    self._add_bash_completion_subparser(subparsers)
    return parser