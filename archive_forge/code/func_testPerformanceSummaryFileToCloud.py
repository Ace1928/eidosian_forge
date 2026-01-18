from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import pprint
import six
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from boto.storage_uri import BucketStorageUri
from gslib import metrics
from gslib import VERSION
from gslib.cs_api_map import ApiSelector
import gslib.exception
from gslib.gcs_json_api import GcsJsonApi
from gslib.metrics import MetricsCollector
from gslib.metrics_tuple import Metric
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SkipForParFile
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('No slice parallelism support for S3.')
def testPerformanceSummaryFileToCloud(self):
    """Tests PerformanceSummary collection in a file-to-cloud transfer."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    file_size = 6
    self.CreateTempFile(tmpdir=tmpdir, contents=b'a' * file_size)
    self.CreateTempFile(tmpdir=tmpdir, contents=b'b' * file_size)
    process_count = 1 if IS_WINDOWS else 2
    with SetBotoConfigForTest([('GSUtil', 'parallel_process_count', str(process_count)), ('GSUtil', 'parallel_thread_count', '3'), ('GSUtil', 'parallel_composite_upload_threshold', '1')]):
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['rsync', tmpdir, suri(bucket_uri)])
        self._CheckParameterValue('Event Category', metrics._GA_PERFSUM_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'FileToCloud', metrics_list)
        self._CheckParameterValue('Parallelism Strategy', 'slice', metrics_list)
        self._CheckParameterValue('Num Processes', str(process_count), metrics_list)
        self._CheckParameterValue('Num Threads', '3', metrics_list)
        self._CheckParameterValue('Provider Types', 'file%2C' + bucket_uri.scheme, metrics_list)
        self._CheckParameterValue('Size of Files/Objects Transferred', 2 * file_size, metrics_list)
        self._CheckParameterValue('Number of Files/Objects Transferred', 2, metrics_list)
        _, _, io_time = self._GetAndCheckAllNumberMetrics(metrics_list)
        if IS_LINUX:
            self.assertGreaterEqual(io_time, 0)