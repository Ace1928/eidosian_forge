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
def test_help_on_subcommand(self):
    if sys.version_info >= (3, 10):
        options_name = 'Options'
    else:
        options_name = 'Optional arguments'
    required = ['.*?^usage: zun create', '.*?^Create a container.', '.*?^' + options_name + ':']
    stdout, stderr = self.shell('help create')
    for r in required:
        self.assertThat(stdout + stderr, matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))