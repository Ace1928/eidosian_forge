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
def testSignUrlWithJSONKeyFileAndObjectGeneration(self):
    """Tests signurl output of a sample object version with JSON keystore."""
    bucket_uri = self.CreateBucket(versioning_enabled=True)
    object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'z')
    cmd = ['signurl', self._GetJSONKsFile(), object_uri.version_specific_uri]
    stdout = self.RunGsUtil(cmd, return_stdout=True)
    self.assertIn('x-goog-credential=' + TEST_EMAIL, stdout)
    self.assertIn('generation=' + object_uri.generation, stdout)