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
@mock.patch('sys.argv', ['zun'])
@mock.patch('sys.stdout', io.StringIO())
@mock.patch('sys.stderr', io.StringIO())
def test_main_noargs(self):
    try:
        zunclient.shell.main()
    except SystemExit:
        self.fail('Unexpected SystemExit')
    self.assertIn('Command-line interface to the OpenStack Zun API', sys.stdout.getvalue())