from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import datetime
import os
import posixpath
import re
import stat
import subprocess
import sys
import time
import gslib
from gslib.commands import ls
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import CaptureStdout
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT1_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT2_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT3_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT4_MD5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_CRC32C
from gslib.tests.util import TEST_ENCRYPTION_CONTENT5_MD5
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY2_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import TEST_ENCRYPTION_KEY3_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY4
from gslib.tests.util import TEST_ENCRYPTION_KEY4_SHA256_B64
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.ls_helper import PrintFullInfoAboutObject
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_bucket_with_Lb(self):
    """Tests ls -Lb."""
    bucket_uri = self.CreateBucket()
    stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
    self.assertIn(suri(bucket_uri), stdout)
    self.assertNotIn('TOTAL:', stdout)
    self.RunGsUtil(['versioning', 'set', 'on', suri(bucket_uri)])
    self.RunGsUtil(['versioning', 'set', 'off', suri(bucket_uri)])
    stdout = self.RunGsUtil(['ls', '-Lb', suri(bucket_uri)], return_stdout=True)
    find_metageneration_re = re.compile('^\\s*Metageneration:\\s+(?P<metageneration_val>.+)$', re.MULTILINE)
    find_time_created_re = re.compile('^\\s*Time created:\\s+(?P<time_created_val>.+)$', re.MULTILINE)
    find_time_updated_re = re.compile('^\\s*Time updated:\\s+(?P<time_updated_val>.+)$', re.MULTILINE)
    metageneration_match = re.search(find_metageneration_re, stdout)
    time_created_match = re.search(find_time_created_re, stdout)
    time_updated_match = re.search(find_time_updated_re, stdout)
    if self.test_api == ApiSelector.XML:
        self.assertIsNone(metageneration_match)
        self.assertIsNone(time_created_match)
        self.assertIsNone(time_updated_match)
    elif self.test_api == ApiSelector.JSON:
        self.assertIsNotNone(metageneration_match)
        self.assertIsNotNone(time_created_match)
        self.assertIsNotNone(time_updated_match)
        time_created = time_created_match.group('time_created_val')
        time_created = time.strptime(time_created, '%a, %d %b %Y %H:%M:%S %Z')
        time_updated = time_updated_match.group('time_updated_val')
        time_updated = time.strptime(time_updated, '%a, %d %b %Y %H:%M:%S %Z')
        self.assertGreater(time_updated, time_created)
        self._AssertBucketPolicyOnly(False, stdout)