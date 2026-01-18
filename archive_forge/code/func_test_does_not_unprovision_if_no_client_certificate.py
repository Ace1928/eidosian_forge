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
def test_does_not_unprovision_if_no_client_certificate(self, mock_remove):
    context_config.create_context_config(self.mock_logger)
    context_config._singleton_config._unprovision_client_cert()
    mock_remove.assert_not_called()