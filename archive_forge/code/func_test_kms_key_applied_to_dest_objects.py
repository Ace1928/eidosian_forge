from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib import command
from gslib.commands import rsync
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3_MD5
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
@SkipForS3('Test uses gs-specific KMS encryption')
def test_kms_key_applied_to_dest_objects(self):
    bucket_uri = self.CreateBucket()
    cloud_container_suri = suri(bucket_uri) + '/foo'
    obj_name = 'bar'
    obj_contents = b'bar'
    tmp_dir = self.CreateTempDir()
    self.CreateTempFile(tmpdir=tmp_dir, file_name=obj_name, contents=obj_contents)
    key_fqn = AuthorizeProjectToUseTestingKmsKey()
    with SetBotoConfigForTest([('GSUtil', 'encryption_key', key_fqn)]):
        self.RunGsUtil(['rsync', tmp_dir, cloud_container_suri])
    with SetBotoConfigForTest([('GSUtil', 'prefer_api', 'json')]):
        stdout = self.RunGsUtil(['ls', '-L', '%s/%s' % (cloud_container_suri, obj_name)], return_stdout=True)
    self.assertRegex(stdout, 'KMS key:\\s+%s' % key_fqn)