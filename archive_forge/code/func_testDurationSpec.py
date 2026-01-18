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
def testDurationSpec(self):
    tests = [('1h', timedelta(hours=1)), ('2d', timedelta(days=2)), ('5D', timedelta(days=5)), ('35s', timedelta(seconds=35)), ('1h', timedelta(hours=1)), ('33', timedelta(hours=33)), ('22m', timedelta(minutes=22)), ('3.7', None), ('27Z', None)]
    for inp, expected in tests:
        try:
            td = gslib.commands.signurl._DurationToTimeDelta(inp)
            self.assertEqual(td, expected)
        except CommandException:
            if expected is not None:
                self.fail('{0} failed to parse')