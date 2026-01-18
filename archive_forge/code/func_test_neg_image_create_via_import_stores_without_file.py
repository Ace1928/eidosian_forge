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
@mock.patch('sys.stdin', autospec=True)
def test_neg_image_create_via_import_stores_without_file(self, mock_stdin, mock_utils_exit):
    expected_msg = '--stores option should only be provided with --file option or stdin for the glance-direct import method.'
    mock_utils_exit.side_effect = self._mock_utils_exit
    mock_stdin.isatty = lambda: True
    my_args = self.base_args.copy()
    my_args.update({'id': 'IMG-01', 'import_method': 'glance-direct', 'stores': 'file1,file2', 'disk_format': 'raw', 'container_format': 'bare'})
    args = self._make_args(my_args)
    with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
        with mock.patch.object(self.gc.images, 'get_stores_info') as mocked_stores_info:
            mocked_stores_info.return_value = self.stores_info_response
            mocked_info.return_value = self.import_info_response
            try:
                test_shell.do_image_create_via_import(self.gc, args)
                self.fail('utils.exit should have been called')
            except SystemExit:
                pass
    mock_utils_exit.assert_called_once_with(expected_msg)