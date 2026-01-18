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
def testConfigValueValidation(self):
    """Tests the validation of potentially PII config values."""
    string_and_bool_categories = ['check_hashes', 'content_language', 'disable_analytics_prompt', 'https_validate_certificates', 'json_api_version', 'parallel_composite_upload_component_size', 'parallel_composite_upload_threshold', 'prefer_api', 'sliced_object_download_component_size', 'sliced_object_download_threshold', 'tab_completion_time_logs', 'token_cache', 'use_magicfile']
    int_categories = ['debug', 'default_api_version', 'http_socket_timeout', 'max_retry_delay', 'num_retries', 'oauth2_refresh_retries', 'parallel_process_count', 'parallel_thread_count', 'resumable_threshold', 'rsync_buffer_lines', 'sliced_object_download_max_components', 'software_update_check_period', 'tab_completion_timeout', 'task_estimation_threshold']
    all_categories = sorted(string_and_bool_categories + int_categories)
    with mock.patch('boto.config.get_value', return_value=None):
        self.assertEqual('', self.collector._ValidateAndGetConfigValues())
    with mock.patch('boto.config.get_value', return_value='invalid string'):
        self.assertEqual(','.join([category + ':INVALID' for category in all_categories]), self.collector._ValidateAndGetConfigValues())
    with mock.patch('boto.config.get_value', return_value='£'):
        self.assertEqual(','.join([category + ':INVALID' for category in all_categories]), self.collector._ValidateAndGetConfigValues())

    def MockValidStrings(section, category):
        if section == 'GSUtil':
            if category == 'check_hashes':
                return 'if_fast_else_skip'
            if category == 'content_language':
                return 'chi'
            if category == 'json_api_version':
                return 'v3'
            if category == 'prefer_api':
                return 'xml'
            if category in ('disable_analytics_prompt', 'use_magicfile', 'tab_completion_time_logs'):
                return 'True'
        if section == 'OAuth2' and category == 'token_cache':
            return 'file_system'
        if section == 'Boto' and category == 'https_validate_certificates':
            return 'True'
        return ''
    with mock.patch('boto.config.get_value', side_effect=MockValidStrings):
        self.assertEqual('check_hashes:if_fast_else_skip,content_language:chi,disable_analytics_prompt:True,https_validate_certificates:True,json_api_version:v3,prefer_api:xml,tab_completion_time_logs:True,token_cache:file_system,use_magicfile:True', self.collector._ValidateAndGetConfigValues())

    def MockValidSmallInts(_, category):
        if category in int_categories:
            return '1999'
        return ''
    with mock.patch('boto.config.get_value', side_effect=MockValidSmallInts):
        self.assertEqual('debug:1999,default_api_version:1999,http_socket_timeout:1999,max_retry_delay:1999,num_retries:1999,oauth2_refresh_retries:1999,parallel_process_count:1999,parallel_thread_count:1999,resumable_threshold:1999,rsync_buffer_lines:1999,sliced_object_download_max_components:1999,software_update_check_period:1999,tab_completion_timeout:1999,task_estimation_threshold:1999', self.collector._ValidateAndGetConfigValues())

    def MockValidLargeInts(_, category):
        if category in int_categories:
            return '2001'
        return ''
    with mock.patch('boto.config.get_value', side_effect=MockValidLargeInts):
        self.assertEqual('debug:INVALID,default_api_version:INVALID,http_socket_timeout:INVALID,max_retry_delay:INVALID,num_retries:INVALID,oauth2_refresh_retries:INVALID,parallel_process_count:INVALID,parallel_thread_count:INVALID,resumable_threshold:2001,rsync_buffer_lines:2001,sliced_object_download_max_components:INVALID,software_update_check_period:INVALID,tab_completion_timeout:INVALID,task_estimation_threshold:2001', self.collector._ValidateAndGetConfigValues())

        def MockNonIntegerValue(_, category):
            if category in int_categories:
                return '10.28'
            return ''
        with mock.patch('boto.config.get_value', side_effect=MockNonIntegerValue):
            self.assertEqual(','.join([category + ':INVALID' for category in int_categories]), self.collector._ValidateAndGetConfigValues())

        def MockDataSizeValue(_, category):
            if category in ('parallel_composite_upload_component_size', 'parallel_composite_upload_threshold', 'sliced_object_download_component_size', 'sliced_object_download_threshold'):
                return '10MiB'
            return ''
        with mock.patch('boto.config.get_value', side_effect=MockDataSizeValue):
            self.assertEqual('parallel_composite_upload_component_size:10485760,parallel_composite_upload_threshold:10485760,sliced_object_download_component_size:10485760,sliced_object_download_threshold:10485760', self.collector._ValidateAndGetConfigValues())