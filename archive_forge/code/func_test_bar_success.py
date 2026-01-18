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
@ddt.data(('--default-is-not-none bar', 'bar'), ('--default_is_not_none bar --default-is-not-none bar', 'bar'), ('--default_is_not_none bar --default_is_not_none bar', 'bar'), ('--default-is-not-none not_bar', 'not_bar'), ('--default_is_not_none None', 'None'))
@ddt.unpack
def test_bar_success(self, options_str, expected_result):
    output = self.shell('bar %s' % options_str)
    parsed_output = output_parser.details(output)
    self.assertEqual({'key': expected_result}, parsed_output)