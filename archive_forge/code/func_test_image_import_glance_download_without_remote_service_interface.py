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
def test_image_import_glance_download_without_remote_service_interface(self):
    args = self._make_args({'id': 'IMG-01', 'uri': None, 'remote_region': 'REGION2', 'remote_image_id': 'IMG-02', 'import_method': 'glance-download'})
    with mock.patch.object(self.gc.images, 'image_import') as mock_import:
        with mock.patch.object(self.gc.images, 'get') as mocked_get:
            with mock.patch.object(self.gc.images, 'get_import_info') as mocked_info:
                mocked_get.return_value = {'status': 'queued', 'container_format': 'bare', 'disk_format': 'raw'}
                mocked_info.return_value = self.import_info_response
                mock_import.return_value = None
                test_shell.do_image_import(self.gc, args)
                mock_import.assert_called_once_with('IMG-01', 'glance-download', uri=None, remote_region='REGION2', remote_image_id='IMG-02', remote_service_interface=None, all_stores=None, allow_failure=True, backend=None, stores=None)