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
@mock.patch('glanceclient.common.utils.print_image')
@mock.patch('glanceclient.v2.shell._validate_backend')
def test_image_import_multiple_stores(self, mocked_utils_print_image, msvb):
    args = self._make_args({'id': 'IMG-02', 'uri': None, 'import_method': 'glance-direct', 'from_create': False, 'stores': 'site1,site2'})
    with mock.patch.object(self.gc.images, 'image_import') as mock_import:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
                mocked_get.return_value = {'status': 'uploading', 'container_format': 'bare', 'disk_format': 'raw'}
                mocked_info.return_value = self.import_info_response
                mock_import.return_value = None
                test_shell.do_image_import(self.gc, args)
                mock_import.assert_called_once_with('IMG-02', 'glance-direct', uri=None, remote_region=None, remote_image_id=None, remote_service_interface=None, all_stores=None, allow_failure=True, stores=['site1', 'site2'], backend=None)