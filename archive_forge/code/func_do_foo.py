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
@cliutils.arg('--default-is-none', '--default_is_none', type=str, metavar='<redefined_metavar>', action='single_alias', help='Default value is None and metavar set.', default=None)
def do_foo(cs, args):
    cliutils.print_dict({'key': args.default_is_none})