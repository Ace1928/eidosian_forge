import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_get_new_file_size(self):
    size = 98304
    file_obj = io.StringIO('X' * size)
    try:
        self.assertEqual(size, utils.get_file_size(file_obj))
        self.assertEqual(0, file_obj.tell())
    finally:
        file_obj.close()