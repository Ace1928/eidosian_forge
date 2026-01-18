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
def test_do_image_create_with_multihash(self):
    self.mock_get_data_file.return_value = io.StringIO()
    try:
        with open(tempfile.mktemp(), 'w+') as f:
            f.write('Some data here')
            f.flush()
            f.seek(0)
            file_name = f.name
        temp_args = {'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare', 'file': file_name, 'progress': False}
        args = self._make_args(temp_args)
        with mock.patch.object(self.gc.images, 'create') as mocked_create:
            with mock.patch.object(self.gc.images, 'get') as mocked_get:
                ignore_fields = ['self', 'access', 'schema']
                expect_image = dict([(field, field) for field in ignore_fields])
                expect_image['id'] = 'pass'
                expect_image['name'] = 'IMG-01'
                expect_image['disk_format'] = 'vhd'
                expect_image['container_format'] = 'bare'
                expect_image['checksum'] = 'fake-checksum'
                expect_image['os_hash_algo'] = 'fake-hash_algo'
                expect_image['os_hash_value'] = 'fake-hash_value'
                mocked_create.return_value = expect_image
                mocked_get.return_value = expect_image
                test_shell.do_image_create(self.gc, args)
                temp_args.pop('file', None)
                mocked_create.assert_called_once_with(**temp_args)
                mocked_get.assert_called_once_with('pass')
                utils.print_dict.assert_called_once_with({'id': 'pass', 'name': 'IMG-01', 'disk_format': 'vhd', 'container_format': 'bare', 'checksum': 'fake-checksum', 'os_hash_algo': 'fake-hash_algo', 'os_hash_value': 'fake-hash_value'})
    finally:
        try:
            os.remove(f.name)
        except Exception:
            pass