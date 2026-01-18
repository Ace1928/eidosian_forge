from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import stat
from gslib.exception import CommandException
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import ReadParallelUploadTrackerFile
from gslib.parallel_tracker_file import ValidateParallelCompositeTrackerData
from gslib.parallel_tracker_file import WriteComponentToParallelUploadTrackerFile
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.storage_url import StorageUrlFromString
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import _HashFilename
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import HashRewriteParameters
from gslib.tracker_file import ReadRewriteTrackerFile
from gslib.tracker_file import WriteRewriteTrackerFile
from gslib.utils import parallelism_framework_util
from gslib.utils.constants import UTF8
def testParallelUploadTrackerFileWithEncryption(self):
    fpath = self.CreateTempFile(file_name='foo')
    random_prefix = '123'
    enc_key = '456'
    objects = [ObjectFromTracker('obj1', '42'), ObjectFromTracker('obj2', '314159')]
    WriteParallelUploadTrackerFile(fpath, random_prefix, objects, encryption_key_sha256=enc_key)
    actual_key, actual_prefix, actual_objects = ReadParallelUploadTrackerFile(fpath, self.logger)
    self.assertEqual(enc_key, actual_key)
    self.assertEqual(random_prefix, actual_prefix)
    self.assertEqual(objects, actual_objects)