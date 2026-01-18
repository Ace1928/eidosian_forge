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
def testRetryableErrorMetadataCollection(self):
    """Tests that retryable errors are collected on JSON metadata operations."""
    if self.test_api != ApiSelector.JSON:
        return unittest.skip('Retryable errors are only collected in JSON')
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'bar')
    self.collector.ga_params[metrics._GA_LABEL_MAP['Command Name']] = 'rsync'
    gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), RetryableErrorsQueue(), self.default_provider)
    gsutil_api.api_client.num_retries = 2
    gsutil_api.api_client.max_retry_wait = 1
    key = object_uri.get_key()
    src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type)
    dst_obj_metadata = apitools_messages.Object(bucket=src_obj_metadata.bucket, name=self.MakeTempName('object'), contentType=src_obj_metadata.contentType)
    with mock.patch.object(http_wrapper, '_MakeRequestNoRetry', side_effect=socket.error()):
        _TryExceptAndPass(gsutil_api.CopyObject, src_obj_metadata, dst_obj_metadata)
    if six.PY2:
        self.assertEqual(self.collector.retryable_errors['SocketError'], 1)
    else:
        self.assertEqual(self.collector.retryable_errors['OSError'], 1)
    with mock.patch.object(http_wrapper, '_MakeRequestNoRetry', side_effect=apitools_exceptions.HttpError('unused', 'unused', 'unused')):
        _TryExceptAndPass(gsutil_api.DeleteObject, bucket_uri.bucket_name, object_uri.object_name)
    self.assertEqual(self.collector.retryable_errors['HttpError'], 1)
    self.assertEqual(self.collector.perf_sum_params.num_retryable_network_errors, 1)
    self.assertEqual(self.collector.perf_sum_params.num_retryable_service_errors, 1)