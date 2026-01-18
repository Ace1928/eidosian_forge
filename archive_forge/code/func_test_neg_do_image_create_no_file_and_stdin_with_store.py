import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
@mock.patch('glanceclient.common.utils.exit')
@mock.patch('os.access')
@mock.patch('sys.stdin', autospec=True)
def test_neg_do_image_create_no_file_and_stdin_with_store(self, mock_stdin, mock_access, mock_utils_exit):
    expected_msg = '--store option should only be provided with --file option or stdin.'
    mock_utils_exit.side_effect = self._mock_utils_exit
    mock_stdin.isatty = lambda: True
    mock_access.return_value = False
    args = self._make_args({'name': 'IMG-01', 'property': ['myprop=myval'], 'file': None, 'store': 'file1', 'container_format': 'bare', 'disk_format': 'qcow2'})
    try:
        test_shell.do_image_create(self.gc, args)
        self.fail('utils.exit should have been called')
    except SystemExit:
        pass
    mock_utils_exit.assert_called_once_with(expected_msg)