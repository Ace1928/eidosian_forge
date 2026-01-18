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
def test_empty_bucket(self):
    bucket_uri = self.CreateBucket()
    self.AssertNObjectsInBucket(bucket_uri, 0)