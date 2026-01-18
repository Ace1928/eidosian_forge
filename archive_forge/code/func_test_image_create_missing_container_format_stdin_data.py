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
@mock.patch('sys.stderr')
def test_image_create_missing_container_format_stdin_data(self, __):
    self.mock_get_data_file.return_value = io.StringIO()
    e = self.assertRaises(exc.CommandError, self._run_command, '--os-image-api-version 2 image-create --disk-format qcow2')
    self.assertEqual('error: Must provide --container-format when using stdin.', e.message)