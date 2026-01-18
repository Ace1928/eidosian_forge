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
@mock.patch.object(subprocess, 'Popen', autospec=True)
def test_executes_custom_provider_command_from_boto_config(self, mock_Popen):
    with SetBotoConfigForTest([('Credentials', 'use_client_certificate', 'True'), ('Credentials', 'cert_provider_command', 'some/path')]):
        with self.assertRaises(ValueError):
            context_config.create_context_config(self.mock_logger)
            mock_Popen.assert_called_once_with(os.path.realpath('some/path'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)