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
@mock.patch(OPEN_TO_PATCH, new_callable=mock.mock_open)
def test_default_provider_no_command_error(self, mock_open, mock_json_load):
    mock_json_load.return_value = DEFAULT_CERT_PROVIDER_FILE_NO_COMMAND
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', None)]):
        context_config.create_context_config(self.mock_logger)
        mock_open.assert_called_with(context_config._DEFAULT_METADATA_PATH)
        self.mock_logger.error.assert_called_once_with('Failed to provision client certificate: Client certificate provider command not found.')