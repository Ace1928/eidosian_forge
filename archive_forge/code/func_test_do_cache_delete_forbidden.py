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
def test_do_cache_delete_forbidden(self):
    with mock.patch('glanceclient.common.utils.print_err') as mock_print_err:
        self._test_cache_delete(forbidden=True)
        mock_print_err.assert_called_once_with("You are not permitted to delete the image 'image1' from cache.")