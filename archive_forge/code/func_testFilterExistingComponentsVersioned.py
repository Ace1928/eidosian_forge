from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import logging
import os
import pyu2f
from apitools.base.py import exceptions as apitools_exceptions
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import CloudApi
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.command import CreateOrGetGsutilLogger
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.storage_url import StorageUrlFromString
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.tests.util import GSMockBucketStorageUri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils import posix_util
from gslib.utils import system_util
from gslib.utils import hashing_helper
from gslib.utils.copy_helper import _CheckCloudHashes
from gslib.utils.copy_helper import _DelegateUploadFileToObject
from gslib.utils.copy_helper import _GetPartitionInfo
from gslib.utils.copy_helper import _SelectUploadCompressionStrategy
from gslib.utils.copy_helper import _SetContentTypeFromFile
from gslib.utils.copy_helper import ExpandUrlToSingleBlr
from gslib.utils.copy_helper import FilterExistingComponents
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import PerformParallelUploadFileToObjectArgs
from gslib.utils.copy_helper import WarnIfMvEarlyDeletionChargeApplies
from six import add_move, MovedModule
from six.moves import mock
def testFilterExistingComponentsVersioned(self):
    """Tests upload with versionined parallel components."""
    mock_api = MockCloudApi()
    bucket_name = self.MakeTempName('bucket')
    mock_api.MockCreateVersionedBucket(bucket_name)
    content_type = 'ContentType'
    storage_class = 'StorageClass'
    tracker_file = self.CreateTempFile(file_name='foo', contents=b'asdf')
    tracker_file_lock = parallelism_framework_util.CreateLock()
    fpath_uploaded_correctly = self.CreateTempFile(file_name='foo1', contents=b'1')
    fpath_uploaded_correctly_url = StorageUrlFromString(str(fpath_uploaded_correctly))
    with open(fpath_uploaded_correctly, 'rb') as f_in:
        fpath_uploaded_correctly_md5 = _CalculateB64EncodedMd5FromContents(f_in)
    object_uploaded_correctly = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_uploaded_correctly, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
    object_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly, object_uploaded_correctly.generation))
    args_uploaded_correctly = PerformParallelUploadFileToObjectArgs(fpath_uploaded_correctly, 0, 1, fpath_uploaded_correctly_url, object_uploaded_correctly_url, object_uploaded_correctly.generation, content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_duplicate = fpath_uploaded_correctly
    fpath_duplicate_url = StorageUrlFromString(str(fpath_duplicate))
    duplicate_uploaded_correctly = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_duplicate, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
    duplicate_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly, duplicate_uploaded_correctly.generation))
    args_duplicate = PerformParallelUploadFileToObjectArgs(fpath_duplicate, 0, 1, fpath_duplicate_url, duplicate_uploaded_correctly_url, duplicate_uploaded_correctly.generation, content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_wrong_contents = self.CreateTempFile(file_name='foo4', contents=b'4')
    fpath_wrong_contents_url = StorageUrlFromString(str(fpath_wrong_contents))
    with open(self.CreateTempFile(contents=b'_'), 'rb') as f_in:
        fpath_wrong_contents_md5 = _CalculateB64EncodedMd5FromContents(f_in)
    object_wrong_contents = mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_wrong_contents, md5Hash=fpath_wrong_contents_md5), contents=b'_')
    wrong_contents_url = StorageUrlFromString('%s://%s/%s#%s' % (self.default_provider, bucket_name, fpath_wrong_contents, object_wrong_contents.generation))
    args_wrong_contents = PerformParallelUploadFileToObjectArgs(fpath_wrong_contents, 0, 1, fpath_wrong_contents_url, wrong_contents_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    dst_args = {fpath_uploaded_correctly: args_uploaded_correctly, fpath_wrong_contents: args_wrong_contents}
    existing_components = [ObjectFromTracker(fpath_uploaded_correctly, object_uploaded_correctly_url.generation), ObjectFromTracker(fpath_duplicate, duplicate_uploaded_correctly_url.generation), ObjectFromTracker(fpath_wrong_contents, wrong_contents_url.generation)]
    bucket_url = StorageUrlFromString('%s://%s' % (self.default_provider, bucket_name))
    components_to_upload, uploaded_components, existing_objects_to_delete = FilterExistingComponents(dst_args, existing_components, bucket_url, mock_api)
    uploaded_components = [i[0] for i in uploaded_components]
    self.assertEqual([args_wrong_contents], components_to_upload)
    self.assertEqual(args_uploaded_correctly.dst_url.url_string, uploaded_components[0].url_string)
    expected_to_delete = [(args_wrong_contents.dst_url.object_name, args_wrong_contents.dst_url.generation), (args_duplicate.dst_url.object_name, args_duplicate.dst_url.generation)]
    for uri in existing_objects_to_delete:
        self.assertTrue((uri.object_name, uri.generation) in expected_to_delete)
    self.assertEqual(len(expected_to_delete), len(existing_objects_to_delete))