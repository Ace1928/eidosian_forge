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
@mock.patch('sys.stdout', autospec=True)
def test_image_download_no_file_arg(self, mocked_stdout, mocked_utils_exit):
    args = self._make_args({'id': '1234', 'file': None, 'progress': False, 'allow_md5_fallback': False})
    mocked_stdout.isatty = lambda: True
    test_shell.do_image_download(self.gc, args)
    mocked_utils_exit.assert_called_once_with('No redirection or local file specified for downloaded image data. Please specify a local file with --file to save downloaded image or redirect output to another source.')