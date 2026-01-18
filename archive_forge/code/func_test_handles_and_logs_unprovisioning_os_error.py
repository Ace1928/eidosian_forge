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
@mock.patch.object(os, 'remove')
def test_handles_and_logs_unprovisioning_os_error(self, mock_remove):
    mock_remove.side_effect = OSError('no')
    context_config.create_context_config(self.mock_logger)
    context_config._singleton_config.client_cert_path = 'some/path'
    context_config._singleton_config._unprovision_client_cert()
    self.mock_logger.error.assert_called_once_with('Failed to remove client certificate: no')