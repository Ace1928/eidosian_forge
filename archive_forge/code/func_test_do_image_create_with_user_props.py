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
@mock.patch('sys.stdin', autospec=True)
def test_do_image_create_with_user_props(self, mock_stdin):
    args = self._make_args({'name': 'IMG-01', 'property': ['myprop=myval'], 'file': None, 'container_format': 'bare', 'disk_format': 'qcow2'})
    with mock.patch.object(self.gc.images, 'create') as mocked_create:
        ignore_fields = ['self', 'access', 'file', 'schema']
        expect_image = dict([(field, field) for field in ignore_fields])
        expect_image['id'] = 'pass'
        expect_image['name'] = 'IMG-01'
        expect_image['myprop'] = 'myval'
        mocked_create.return_value = expect_image
        mock_stdin.isatty = lambda: True
        test_shell.do_image_create(self.gc, args)
        mocked_create.assert_called_once_with(name='IMG-01', myprop='myval', container_format='bare', disk_format='qcow2')
        utils.print_dict.assert_called_once_with({'id': 'pass', 'name': 'IMG-01', 'myprop': 'myval'})