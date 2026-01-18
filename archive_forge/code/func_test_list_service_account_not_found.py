from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_list_service_account_not_found(self):
    service_account = 'service-account-DNE@gmail.com'
    stderr = self.RunGsUtil(['hmac', 'list', '-u', service_account], return_stderr=True, expected_status=1)
    if self._use_gcloud_storage:
        self.assertIn("HTTPError 404: Service Account '{}' not found.".format(service_account), stderr)
    else:
        self.assertIn("Service Account '%s' not found." % service_account, stderr)