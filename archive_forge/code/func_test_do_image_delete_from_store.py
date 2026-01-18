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
def test_do_image_delete_from_store(self):
    args = argparse.Namespace(id='image1', store='store1')
    with mock.patch.object(self.gc.images, 'delete_from_store') as mocked_delete:
        test_shell.do_stores_delete(self.gc, args)
        mocked_delete.assert_called_once_with('store1', 'image1')