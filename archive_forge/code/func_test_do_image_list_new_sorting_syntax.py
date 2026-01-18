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
def test_do_image_list_new_sorting_syntax(self):
    input = {'limit': None, 'page_size': 18, 'visibility': True, 'member_status': 'Fake', 'owner': 'test', 'checksum': 'fake_checksum', 'tag': 'fake tag', 'properties': [], 'sort': 'name:desc,size:asc', 'sort_key': [], 'sort_dir': [], 'verbose': False, 'include_stores': False, 'os_hash_value': None, 'os_hidden': False}
    args = self._make_args(input)
    with mock.patch.object(self.gc.images, 'list') as mocked_list:
        mocked_list.return_value = {}
        test_shell.do_image_list(self.gc, args)
        exp_img_filters = {'owner': 'test', 'member_status': 'Fake', 'visibility': True, 'checksum': 'fake_checksum', 'tag': 'fake tag', 'os_hidden': False}
        mocked_list.assert_called_once_with(page_size=18, sort='name:desc,size:asc', filters=exp_img_filters)
        utils.print_list.assert_called_once_with({}, ['ID', 'Name'])