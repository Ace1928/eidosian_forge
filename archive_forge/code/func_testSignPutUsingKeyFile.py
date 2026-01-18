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
def testSignPutUsingKeyFile(self):
    """Tests the _GenSignedUrl function with a PUT method using Key file."""
    expected = sigs.TEST_SIGN_PUT_SIG
    duration = timedelta(seconds=3600)
    with SetBotoConfigForTest([('Credentials', 'gs_host', 'storage.googleapis.com')]):
        signed_url = gslib.commands.signurl._GenSignedUrl(self.key, api=None, use_service_account=False, provider='gs', client_id=self.client_email, method='RESUMABLE', gcs_path='test/test.txt', duration=duration, logger=self.logger, region='us-east', content_type='')
    self.assertEqual(expected, signed_url)