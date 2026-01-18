import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
def test_help_no_options(self):
    required = ['.*?^usage: ', '.*?^\\s+stop\\s+Stop specified container.', '.*?^See "zun help COMMAND" for help on a specific command']
    stdout, stderr = self.shell('')
    for r in required:
        self.assertThat(stdout + stderr, matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))