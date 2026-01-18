from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ast
import base64
import binascii
import datetime
import gzip
import logging
import os
import pickle
import pkgutil
import random
import re
import stat
import string
import sys
import threading
from unittest import mock
from apitools.base.py import exceptions as apitools_exceptions
import boto
from boto import storage_uri
from boto.exception import ResumableTransferDisposition
from boto.exception import StorageResponseError
from boto.storage_uri import BucketStorageUri
from gslib import command
from gslib import exception
from gslib import name_expansion
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import _DEFAULT_DOWNLOAD_CHUNK_SIZE
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import InvalidUrlError
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import NotParallelizable
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import HAS_GS_PORT
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import KmsTestingResources
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
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.ui_controller import BytesToFixedWidthString
from gslib.utils import hashing_helper
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import GetTrackerFilePath
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.copy_helper import TrackerFileType
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import ValidatePOSIXMode
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.text_util import get_random_ascii_chars
from gslib.utils.unit_util import EIGHT_MIB
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from gslib.utils import shim_util
import six
from six.moves import http_client
from six.moves import range
from six.moves import xrange
@SkipForS3('No compressed transport encoding support for S3.')
@SkipForXML('No compressed transport encoding support for the XML API.')
@SequentialAndParallelTransfer
def test_gzip_transport_encoded_upload_and_download(self):
    """Test gzip encoded files upload correctly.

    This checks that files are not tagged with a gzip content encoding and
    that the contents of the files are uncompressed in GCS. This test uses the
    -j flag to target specific extensions.
    """

    def _create_test_data():
        """Setup the bucket and local data to test with.

      Returns:
        Triplet containing the following values:
          bucket_uri: String URI of cloud storage bucket to upload mock data
                      to.
          tmpdir: String, path of a temporary directory to write mock data to.
          local_uris: Tuple of three strings; each is the file path to a file
                      containing mock data.
      """
        bucket_uri = self.CreateBucket()
        contents = b'x' * 10000
        tmpdir = self.CreateTempDir()
        local_uris = []
        for filename in ('test.html', 'test.js', 'test.txt'):
            local_uris.append(self.CreateTempFile(file_name=filename, tmpdir=tmpdir, contents=contents))
        return (bucket_uri, tmpdir, local_uris)

    def _upload_test_data(tmpdir, bucket_uri):
        """Upload local test data.

      Args:
        tmpdir: String, path of a temporary directory to write mock data to.
        bucket_uri: String URI of cloud storage bucket to upload mock data to.

      Returns:
        stderr: String output from running the gsutil command to upload mock
                  data.
      """
        if self._use_gcloud_storage:
            extension_list_string = 'js,html'
        else:
            extension_list_string = 'js, html'
        stderr = self.RunGsUtil(['-D', 'cp', '-j', extension_list_string, os.path.join(tmpdir, 'test*'), suri(bucket_uri)], return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, 3)
        return stderr

    def _assert_sent_compressed(local_uris, stderr):
        """Ensure the correct files were marked for compression.

      Args:
        local_uris: Tuple of three strings; each is the file path to a file
                    containing mock data.
        stderr: String output from running the gsutil command to upload mock
                data.
      """
        local_uri_html, local_uri_js, local_uri_txt = local_uris
        assert_base_string = 'Using compressed transport encoding for file://{}.'
        self.assertIn(assert_base_string.format(local_uri_html), stderr)
        self.assertIn(assert_base_string.format(local_uri_js), stderr)
        self.assertNotIn(assert_base_string.format(local_uri_txt), stderr)

    def _assert_stored_uncompressed(bucket_uri, contents=b'x' * 10000):
        """Ensure the files are not compressed when they are stored in the bucket.

      Args:
        bucket_uri: String with URI for bucket containing uploaded test data.
        contents: Byte string that are stored in each file in the bucket.
      """
        local_uri_html = suri(bucket_uri, 'test.html')
        local_uri_js = suri(bucket_uri, 'test.js')
        local_uri_txt = suri(bucket_uri, 'test.txt')
        fpath4 = self.CreateTempFile()
        for uri in (local_uri_html, local_uri_js, local_uri_txt):
            stdout = self.RunGsUtil(['stat', uri], return_stdout=True)
            self.assertNotRegex(stdout, 'Content-Encoding:\\s+gzip')
            self.RunGsUtil(['cp', uri, suri(fpath4)])
            with open(fpath4, 'rb') as f:
                self.assertEqual(f.read(), contents)
    bucket_uri, tmpdir, local_uris = _create_test_data()
    stderr = _upload_test_data(tmpdir, bucket_uri)
    _assert_sent_compressed(local_uris, stderr)
    _assert_stored_uncompressed(bucket_uri)