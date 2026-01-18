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
def test_do_cache_clear_unsupported(self):
    with mock.patch('glanceclient.common.utils.print_err') as mock_print_err:
        self._test_cache_clear(supported=False)
        mock_print_err.assert_called_once_with("'HTTP HTTPNotImplemented': Unable to delete image(s) from cache.")