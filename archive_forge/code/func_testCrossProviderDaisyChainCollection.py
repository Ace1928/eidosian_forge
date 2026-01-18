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
@unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
def testCrossProviderDaisyChainCollection(self):
    """Tests the collection of daisy-chain operations."""
    s3_bucket = self.CreateBucket(provider='s3')
    gs_bucket = self.CreateBucket(provider='gs')
    unused_s3_key = self.CreateObject(bucket_uri=s3_bucket, contents=b'foo')
    gs_key = self.CreateObject(bucket_uri=gs_bucket, contents=b'bar')
    metrics_list = self._RunGsUtilWithAnalyticsOutput(['rsync', suri(s3_bucket), suri(gs_bucket)])
    self._CheckParameterValue('Event Action', 'CloudToCloud%2CDaisyChain', metrics_list)
    self._CheckParameterValue('Provider Types', 'gs%2Cs3', metrics_list)
    metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', suri(gs_key), suri(s3_bucket)])
    self._CheckParameterValue('Event Action', 'CloudToCloud%2CDaisyChain', metrics_list)
    self._CheckParameterValue('Provider Types', 'gs%2Cs3', metrics_list)