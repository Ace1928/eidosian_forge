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
@ddt.data('--default-is-none foo --default-is-none bar', '--default-is-none foo --default_is_none bar', '--default-is-none foo --default_is_none FOO')
def test_foo_error(self, options_str):
    self.assertRaises(matchers.MismatchError, self.shell, 'foo %s' % options_str)