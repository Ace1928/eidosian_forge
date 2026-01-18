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
@mock.patch('glanceclient.v2.shell.do_image_import')
@mock.patch('glanceclient.v2.shell.do_image_stage')
@mock.patch('sys.stdin', autospec=True)
def test_do_image_create_via_import_with_web_download_with_stores(self, mock_stdin, mock_do_image_stage, mock_do_image_import):
    temp_args = {'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare', 'uri': 'http://example.com/image.qcow', 'import_method': 'web-download', 'progress': False, 'stores': 'file1,file2'}
    tmp2_args = {'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare', 'uri': 'http://example.com/image.qcow', 'import_method': 'web-download', 'progress': False}
    args = self._make_args(temp_args)
    with mock.patch.object(self.gc.images, 'create') as mocked_create:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
                with mock.patch.object(self.gc.images, 'get_stores_info') as m_stores_info:
                    ignore_fields = ['self', 'access', 'schema']
                    expect_image = dict([(field, field) for field in ignore_fields])
                    expect_image['id'] = 'pass'
                    expect_image['name'] = 'IMG-01'
                    expect_image['disk_format'] = 'vhd'
                    expect_image['container_format'] = 'bare'
                    expect_image['status'] = 'queued'
                    mocked_create.return_value = expect_image
                    mocked_get.return_value = expect_image
                    mocked_info.return_value = self.import_info_response
                    m_stores_info.return_value = self.stores_info_response
                    mock_stdin.isatty = lambda: True
                    test_shell.do_image_create_via_import(self.gc, args)
                    mock_do_image_stage.assert_not_called()
                    mock_do_image_import.assert_called_once()
                    mocked_create.assert_called_once_with(**tmp2_args)
                    mocked_get.assert_called_with('pass')
                    utils.print_dict.assert_called_with({'id': 'pass', 'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare', 'status': 'queued'})