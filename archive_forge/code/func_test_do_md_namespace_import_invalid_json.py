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
def test_do_md_namespace_import_invalid_json(self):
    args = self._make_args({'file': 'test'})
    mock_read = mock.Mock(return_value='Invalid')
    mock_file = mock.Mock(read=mock_read)
    utils.get_data_file = mock.Mock(return_value=mock_file)
    self.assertRaises(SystemExit, test_shell.do_md_namespace_import, self.gc, args)