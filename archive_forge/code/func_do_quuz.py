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
@staticmethod
@cliutils.arg('--list-like', '--list_like', nargs='*', action='single_alias', help='Default value is None, metavar not set and result is list.', default=None)
def do_quuz(cs, args):
    cliutils.print_dict({'key': args.list_like})