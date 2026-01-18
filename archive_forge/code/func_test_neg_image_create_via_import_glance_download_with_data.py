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
def test_neg_image_create_via_import_glance_download_with_data(self, mock_stdin, mock_utils_exit):
    expected_msg = 'You cannot pass data via stdin with the glance-download import method.'
    my_args = self.base_args.copy()
    my_args['import_method'] = 'glance-download'
    my_args['remote_region'] = 'REGION2'
    my_args['remote_image_id'] = 'IMG2'
    args = self._make_args(my_args)
    mock_stdin.isatty = lambda: False
    mock_utils_exit.side_effect = self._mock_utils_exit
    with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
        mocked_info.return_value = self.import_info_response
        try:
            test_shell.do_image_create_via_import(self.gc, args)
            self.fail('utils.exit should have been called')
        except SystemExit:
            pass
    mock_utils_exit.assert_called_once_with(expected_msg)