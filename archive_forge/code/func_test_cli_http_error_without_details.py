import io
import sys
from unittest import mock
from keystoneauth1 import exceptions
import testtools
from aodhclient import shell
@mock.patch('sys.stderr', io.StringIO())
def test_cli_http_error_without_details(self):
    shell.AodhShell().clean_up(None, None, exceptions.HttpError('foo'))
    stderr_lines = sys.stderr.getvalue().splitlines()
    self.assertEqual(0, len(stderr_lines))