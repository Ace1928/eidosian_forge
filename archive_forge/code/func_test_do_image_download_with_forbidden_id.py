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
@mock.patch('sys.stdout', autospec=True)
@mock.patch.object(utils, 'print_err')
def test_do_image_download_with_forbidden_id(self, mocked_print_err, mocked_stdout):
    args = self._make_args({'id': 'IMG-01', 'file': None, 'progress': False, 'allow_md5_fallback': False})
    mocked_stdout.isatty = lambda: False
    with mock.patch.object(self.gc.images, 'data') as mocked_data:
        mocked_data.side_effect = exc.HTTPForbidden
        try:
            test_shell.do_image_download(self.gc, args)
            self.fail('Exit not called')
        except SystemExit:
            pass
        self.assertEqual(1, mocked_data.call_count)
        self.assertEqual(1, mocked_print_err.call_count)