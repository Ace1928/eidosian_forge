import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
@mock.patch.object(builtins, 'open', autospec=True)
def test_handle_json_or_file_arg_file_fail(self, mock_open):
    mock_open.return_value.__enter__.side_effect = IOError
    with tempfile.NamedTemporaryFile(mode='w') as f:
        self.assertRaisesRegex(exc.InvalidAttribute, 'from file', utils.handle_json_or_file_arg, f.name)
        mock_open.assert_called_once_with(f.name, 'r')