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
def test_do_image_show(self):
    args = self._make_args({'id': 'pass', 'page_size': 18, 'human_readable': False, 'max_column_width': 120})
    with mock.patch.object(self.gc.images, 'get') as mocked_list:
        ignore_fields = ['self', 'access', 'file', 'schema']
        expect_image = dict([(field, field) for field in ignore_fields])
        expect_image['id'] = 'pass'
        expect_image['size'] = 1024
        mocked_list.return_value = expect_image
        test_shell.do_image_show(self.gc, args)
        mocked_list.assert_called_once_with('pass')
        utils.print_dict.assert_called_once_with({'id': 'pass', 'size': 1024}, max_column_width=120)