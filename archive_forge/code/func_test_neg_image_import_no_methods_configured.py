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
def test_neg_image_import_no_methods_configured(self, mock_utils_exit):
    expected_msg = "Import method 'glance-direct' is not valid for this cloud. Valid values can be retrieved with import-info command."
    mock_utils_exit.side_effect = self._mock_utils_exit
    args = self._make_args({'id': 'IMG-01', 'import_method': 'glance-direct', 'uri': None})
    with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
        mocked_info.return_value = {'import-methods': {'value': []}}
        try:
            test_shell.do_image_import(self.gc, args)
            self.fail('utils.exit should have been called')
        except SystemExit:
            pass
    mock_utils_exit.assert_called_once_with(expected_msg)