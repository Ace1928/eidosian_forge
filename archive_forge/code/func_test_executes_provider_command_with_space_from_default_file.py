from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import subprocess
from unittest import mock
import six
from gslib import context_config
from gslib import exception
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
@mock.patch('os.path.exists', new=mock.Mock(return_value=True))
@mock.patch.object(json, 'load', autospec=True)
@mock.patch.object(subprocess, 'Popen', autospec=True)
@mock.patch(OPEN_TO_PATCH, new_callable=mock.mock_open)
def test_executes_provider_command_with_space_from_default_file(self, mock_open, mock_Popen, mock_json_load):
    mock_json_load.side_effect = [DEFAULT_CERT_PROVIDER_FILE_CONTENTS_WITH_SPACE]
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', None)]):
        with self.assertRaises(ValueError):
            context_config.create_context_config(self.mock_logger)
            mock_open.assert_called_with(context_config._DEFAULT_METADATA_PATH)
            mock_Popen.assert_called_once_with(os.path.realpath('cert helper'), '--print_certificate', '--with_passphrase')