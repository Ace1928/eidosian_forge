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
@mock.patch.object(subprocess, 'Popen')
def test_does_not_provision_if_use_client_certificate_not_true(self, mock_Popen):
    context_config.create_context_config(self.mock_logger)
    mock_Popen.assert_not_called()