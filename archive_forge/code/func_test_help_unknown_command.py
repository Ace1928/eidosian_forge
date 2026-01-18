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
def test_help_unknown_command(self):
    self.assertRaises(exceptions.CommandError, self.shell, 'help foofoo')