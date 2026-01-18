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
def testRetryableErrorMediaCollection(self):
    """Tests that retryable errors are collected on JSON media operations."""
    if self.test_api != ApiSelector.JSON:
        return unittest.skip('Retryable errors are only collected in JSON')
    boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB))]
    bucket_uri = self.CreateBucket()
    halt_size = START_CALLBACK_PER_BYTES * 2
    fpath = self.CreateTempFile(contents=b'a' * halt_size)
    test_callback_file = self.CreateTempFile(contents=pickle.dumps(_ResumableUploadRetryHandler(5, apitools_exceptions.BadStatusCodeError, ('unused', 'unused', 'unused'))))
    with SetBotoConfigForTest(boto_config_for_test):
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
        self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'BadStatusCodeError', metrics_list)
        self._CheckParameterValue('Retryable Errors', '1', metrics_list)
        self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)
    test_callback_file = self.CreateTempFile(contents=pickle.dumps(_JSONForceHTTPErrorCopyCallbackHandler(5, 404)))
    with SetBotoConfigForTest(boto_config_for_test):
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
        self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'ResumableUploadStartOverException', metrics_list)
        self._CheckParameterValue('Retryable Errors', '1', metrics_list)
        self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)
    test_callback_file = self.CreateTempFile(contents=pickle.dumps(_JSONForceHTTPErrorCopyCallbackHandler(5, 404)))
    with SetBotoConfigForTest(boto_config_for_test):
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['-m', 'cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
        self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'ResumableUploadStartOverException', metrics_list)
        self._CheckParameterValue('Retryable Errors', '1', metrics_list)
        self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)