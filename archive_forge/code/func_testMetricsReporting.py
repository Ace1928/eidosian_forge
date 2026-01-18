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
@mock.patch('time.time', new=mock.MagicMock(return_value=0))
def testMetricsReporting(self):
    """Tests the subprocess creation by Popen in metrics.py."""
    popen_mock = self._StartObjectPatch(subprocess, 'Popen')
    metrics_file = tempfile.NamedTemporaryFile()
    metrics_file.close()
    temp_file_mock = self._StartObjectPatch(tempfile, 'NamedTemporaryFile')
    temp_file_mock.return_value = open(metrics_file.name, 'wb')
    self.collector.ReportMetrics()
    self.assertEqual(0, popen_mock.call_count)
    _LogAllTestMetrics()
    metrics.Shutdown()
    call_list = popen_mock.call_args_list
    self.assertEqual(1, len(call_list))
    args = call_list[0]
    self.assertIn('PYTHONPATH', args[1]['env'])
    missing_paths = set(sys.path) - set(args[1]['env']['PYTHONPATH'].split(os.pathsep))
    self.assertEqual(set(), missing_paths)
    with open(metrics_file.name, 'rb') as metrics_file:
        reported_metrics = pickle.load(metrics_file)
    self.assertEqual(COMMAND_AND_ERROR_TEST_METRICS, set(reported_metrics))