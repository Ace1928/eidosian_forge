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
@ddt.data('--default-is-not-none foo --default-is-not-none bar', '--default-is-not-none foo --default_is_not_none bar', '--default-is-not-none bar --default_is_not_none BAR')
def test_bar_error(self, options_str):
    self.assertRaises(matchers.MismatchError, self.shell, 'bar %s' % options_str)