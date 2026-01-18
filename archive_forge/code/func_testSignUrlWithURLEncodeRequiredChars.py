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
def testSignUrlWithURLEncodeRequiredChars(self):
    objs = ['gs://example.org/test 1', 'gs://example.org/test/test 2', 'gs://example.org/Аудиоарi хив']
    expected_partial_urls = ['https://storage.googleapis.com/example.org/test%201?x-goog-signature=', 'https://storage.googleapis.com/example.org/test/test%202?x-goog-signature=', 'https://storage.googleapis.com/example.org/%D0%90%D1%83%D0%B4%D0%B8%D0%BE%D0%B0%D1%80i%20%D1%85%D0%B8%D0%B2?x-goog-signature=']
    self.assertEqual(len(objs), len(expected_partial_urls))
    cmd_args = ['signurl', '-m', 'PUT', '-p', 'notasecret', '-r', 'us', self._GetKsFile()]
    cmd_args.extend(objs)
    stdout = self.RunGsUtil(cmd_args, return_stdout=True)
    lines = stdout.split('\n')
    self.assertEqual(len(lines), len(objs) + 2)
    lines = lines[1:]
    for obj, line, partial_url in zip(objs, lines, expected_partial_urls):
        self.assertIn(obj, line)
        self.assertIn(partial_url, line)
        self.assertIn('x-goog-credential=test.apps.googleusercontent.com', line)
    self.assertIn('%2Fus%2F', stdout)