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
def test_do_member_get(self):
    args = self._make_args({'image_id': 'IMG-01', 'member_id': 'MEM-01'})
    with mock.patch.object(self.gc.image_members, 'get') as mock_get:
        mock_get.return_value = {}
        test_shell.do_member_get(self.gc, args)
        mock_get.assert_called_once_with('IMG-01', 'MEM-01')
        utils.print_dict.assert_called_once_with({})