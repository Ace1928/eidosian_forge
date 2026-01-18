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
@mock.patch('os.access')
@mock.patch('sys.stdin', autospec=True)
def test_image_create_via_import_no_method_passing_file(self, mock_stdin, mock_access, mock_do_stage, mock_do_import):
    """Backward compat -> handle this like a glance-direct"""
    mock_stdin.isatty = lambda: True
    self.mock_get_data_file.return_value = io.StringIO()
    mock_access.return_value = True
    my_args = self.base_args.copy()
    my_args['file'] = 'fake-image-file.browncow'
    args = self._make_args(my_args)
    with mock.patch.object(self.gc.images, 'create') as mocked_create:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
                ignore_fields = ['self', 'access', 'schema']
                expect_image = dict([(field, field) for field in ignore_fields])
                expect_image['id'] = 'via-file'
                expect_image['name'] = 'Mortimer'
                expect_image['disk_format'] = 'raw'
                expect_image['container_format'] = 'bare'
                mocked_create.return_value = expect_image
                mocked_get.return_value = expect_image
                mocked_info.return_value = self.import_info_response
                test_shell.do_image_create_via_import(self.gc, args)
                mocked_create.assert_called_once()
                mock_do_stage.assert_called_once()
                mock_do_import.assert_called_once()
                mocked_get.assert_called_with('via-file')
                utils.print_dict.assert_called_with({'id': 'via-file', 'name': 'Mortimer', 'disk_format': 'raw', 'container_format': 'bare'})