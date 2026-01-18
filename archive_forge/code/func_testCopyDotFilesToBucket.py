from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
def testCopyDotFilesToBucket(self):
    dst_bucket_uri = self.CreateBucket()
    src_dir = self.CreateTempDir(test_files=['foo'])
    object_named_dot = suri(dst_bucket_uri) + '/.'
    object_named_dotdot = suri(dst_bucket_uri) + '/..'
    for object_name in (object_named_dot, object_named_dotdot):
        try:
            self.RunCommand('cp', [os.path.join(src_dir, 'foo'), object_name])
            self.fail('Expected InvalidUrlError for %s' % object_name)
        except InvalidUrlError:
            pass