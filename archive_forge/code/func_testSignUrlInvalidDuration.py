from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
from datetime import timedelta
import os
import pkgutil
import boto
import gslib.commands.signurl
from gslib.commands.signurl import HAVE_OPENSSL
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.iamcredentials_api import IamcredentailsApi
from gslib.impersonation_credentials import ImpersonationCredentials
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import (SkipForS3, SkipForXML)
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
import gslib.tests.signurl_signatures as sigs
from oauth2client import client
from oauth2client.service_account import ServiceAccountCredentials
from six import add_move, MovedModule
from six.moves import mock
def testSignUrlInvalidDuration(self):
    """Tests signurl fails with out of bounds value for valid duration."""
    if self._use_gcloud_storage:
        expected_status = 2
    else:
        expected_status = 1
    stderr = self.RunGsUtil(['signurl', '-d', '123d', 'ks_file', 'gs://uri'], return_stderr=True, expected_status=expected_status)
    if self._use_gcloud_storage:
        self.assertIn('value must be less than or equal to 7d', stderr)
    else:
        self.assertIn('CommandException: Max valid duration allowed is 7 days', stderr)