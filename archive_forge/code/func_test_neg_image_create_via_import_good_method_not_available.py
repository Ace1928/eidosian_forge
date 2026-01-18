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
def test_neg_image_create_via_import_good_method_not_available(self, mock_stdin, mock_utils_exit):
    """Make sure the good method names aren't hard coded somewhere"""
    expected_msg = "Import method 'glance-direct' is not valid for this cloud. Valid values can be retrieved with import-info command."
    my_args = self.base_args.copy()
    my_args['import_method'] = 'glance-direct'
    args = self._make_args(my_args)
    mock_stdin.isatty = lambda: True
    mock_utils_exit.side_effect = self._mock_utils_exit
    my_import_info_response = deepcopy(self.import_info_response)
    my_import_info_response['import-methods']['value'] = ['bad-bad-method']
    with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
        mocked_info.return_value = my_import_info_response
        try:
            test_shell.do_image_create_via_import(self.gc, args)
            self.fail('utils.exit should have been called')
        except SystemExit:
            pass
    mock_utils_exit.assert_called_once_with(expected_msg)