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
@mock.patch.object(utils, 'exit')
def test_do_image_delete_from_store_not_found(self, mocked_utils_exit):
    args = argparse.Namespace(id='image1', store='store1')
    with mock.patch.object(self.gc.images, 'delete_from_store') as mocked_delete:
        mocked_delete.side_effect = exc.HTTPNotFound
        test_shell.do_stores_delete(self.gc, args)
        self.assertEqual(1, mocked_delete.call_count)
        mocked_utils_exit.assert_called_once_with('Multi Backend support is not enabled or Image/store not found.')