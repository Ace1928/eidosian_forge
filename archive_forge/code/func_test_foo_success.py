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
@ddt.data(('--default-is-none foo', 'foo'), ('--default-is-none foo --default-is-none foo', 'foo'), ('--default-is-none foo --default_is_none foo', 'foo'), ('--default_is_none None', 'None'))
@ddt.unpack
def test_foo_success(self, options_str, expected_result):
    output = self.shell('foo %s' % options_str)
    parsed_output = output_parser.details(output)
    self.assertEqual({'key': expected_result}, parsed_output)