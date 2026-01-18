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
def test_RewriteTrackerFile(self):
    """Tests Rewrite tracker file functions."""
    tracker_file_name = GetRewriteTrackerFilePath('bk1', 'obj1', 'bk2', 'obj2', self.test_api)
    DeleteTrackerFile(tracker_file_name)
    src_obj_metadata = apitools_messages.Object(bucket='bk1', name='obj1', etag='etag1', md5Hash='12345')
    src_obj2_metadata = apitools_messages.Object(bucket='bk1', name='obj1', etag='etag2', md5Hash='67890')
    dst_obj_metadata = apitools_messages.Object(bucket='bk2', name='obj2')
    rewrite_token = 'token1'
    self.assertIsNone(ReadRewriteTrackerFile(tracker_file_name, src_obj_metadata))
    rewrite_params_hash = HashRewriteParameters(src_obj_metadata, dst_obj_metadata, 'full')
    WriteRewriteTrackerFile(tracker_file_name, rewrite_params_hash, rewrite_token)
    self.assertEqual(ReadRewriteTrackerFile(tracker_file_name, rewrite_params_hash), rewrite_token)
    rewrite_params_hash2 = HashRewriteParameters(src_obj2_metadata, dst_obj_metadata, 'full')
    self.assertIsNone(ReadRewriteTrackerFile(tracker_file_name, rewrite_params_hash2))
    DeleteTrackerFile(tracker_file_name)