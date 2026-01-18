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
def test_do_member_list(self):
    args = self._make_args({'image_id': 'IMG-01'})
    with mock.patch.object(self.gc.image_members, 'list') as mocked_list:
        mocked_list.return_value = {}
        test_shell.do_member_list(self.gc, args)
        mocked_list.assert_called_once_with('IMG-01')
        columns = ['Image ID', 'Member ID', 'Status']
        utils.print_list.assert_called_once_with({}, columns)