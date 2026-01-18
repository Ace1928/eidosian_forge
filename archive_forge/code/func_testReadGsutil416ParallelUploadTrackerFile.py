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
def testReadGsutil416ParallelUploadTrackerFile(self):
    """Tests the parallel upload tracker file format prior to gsutil 4.17."""
    random_prefix = '123'
    objects = ['obj1', '42', 'obj2', '314159']
    contents = '\n'.join([random_prefix] + objects) + '\n'
    fpath = self.CreateTempFile(file_name='foo', contents=contents.encode(UTF8))
    expected_objects = [ObjectFromTracker(objects[2 * i], objects[2 * i + 1]) for i in range(0, len(objects) // 2)]
    _, actual_prefix, actual_objects = ReadParallelUploadTrackerFile(fpath, self.logger)
    self.assertEqual(random_prefix, actual_prefix)
    self.assertEqual(expected_objects, actual_objects)