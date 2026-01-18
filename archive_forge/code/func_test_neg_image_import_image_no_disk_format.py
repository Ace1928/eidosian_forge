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
def test_neg_image_import_image_no_disk_format(self, mock_utils_exit):
    expected_msg = "The 'container_format' and 'disk_format' properties must be set on an image before it can be imported."
    mock_utils_exit.side_effect = self._mock_utils_exit
    args = self._make_args({'id': 'IMG-01', 'import_method': 'web-download', 'uri': 'http://joes-image-shack.com/funky.qcow2'})
    with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            mocked_get.return_value = {'status': 'uploading', 'container_format': 'bare'}
            mocked_info.return_value = self.import_info_response
            try:
                test_shell.do_image_import(self.gc, args)
                self.fail('utils.exit should have been called')
            except SystemExit:
                pass
    mock_utils_exit.assert_called_once_with(expected_msg)