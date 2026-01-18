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
def testSignUrlWithWildcard(self):
    objs = ['test1', 'test2', 'test3']
    obj_urls = []
    bucket = self.CreateBucket()
    for obj_name in objs:
        obj_urls.append(self.CreateObject(bucket_uri=bucket, object_name=obj_name, contents=b''))
    stdout = self.RunGsUtil(['signurl', '-p', 'notasecret', self._GetKsFile(), suri(bucket) + '/*'], return_stdout=True)
    self.assertEqual(len(stdout.split('\n')), 5)
    for obj_url in obj_urls:
        self.assertIn(suri(obj_url), stdout)