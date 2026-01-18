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
@mock.patch.object(http_wrapper, 'HandleExceptionsAndRebuildHttpConnections')
def testRetryableErrorCollection(self, mock_default_retry):
    """Tests the collection of a retryable error in the retry function."""
    mock_queue = RetryableErrorsQueue()
    value_error_retry_args = http_wrapper.ExceptionRetryArgs(None, None, ValueError(), None, None, None)
    socket_error_retry_args = http_wrapper.ExceptionRetryArgs(None, None, socket.error(), None, None, None)
    metadata_retry_func = LogAndHandleRetries(is_data_transfer=False, status_queue=mock_queue)
    media_retry_func = LogAndHandleRetries(is_data_transfer=True, status_queue=mock_queue)
    metadata_retry_func(value_error_retry_args)
    self.assertEqual(self.collector.retryable_errors['ValueError'], 1)
    metadata_retry_func(value_error_retry_args)
    self.assertEqual(self.collector.retryable_errors['ValueError'], 2)
    metadata_retry_func(socket_error_retry_args)
    if six.PY2:
        self.assertEqual(self.collector.retryable_errors['SocketError'], 1)
    else:
        self.assertEqual(self.collector.retryable_errors['OSError'], 1)
    _TryExceptAndPass(media_retry_func, value_error_retry_args)
    _TryExceptAndPass(media_retry_func, socket_error_retry_args)
    self.assertEqual(self.collector.retryable_errors['ValueError'], 3)
    if six.PY2:
        self.assertEqual(self.collector.retryable_errors['SocketError'], 2)
    else:
        self.assertEqual(self.collector.retryable_errors['OSError'], 2)