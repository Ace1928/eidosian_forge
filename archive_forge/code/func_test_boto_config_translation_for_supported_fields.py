from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
@mock.patch.object(boto_util, 'UsingGsHmac', return_value=True)
def test_boto_config_translation_for_supported_fields(self, _):
    with _mock_boto_config({'Credentials': {'aws_access_key_id': 'AWS_ACCESS_KEY_ID_value', 'aws_secret_access_key': 'AWS_SECRET_ACCESS_KEY_value', 'gs_access_key_id': 'CLOUDSDK_STORAGE_GS_XML_ACCESS_KEY_ID_value', 'gs_secret_access_key': 'CLOUDSDK_STORAGE_GS_XML_SECRET_ACCESS_KEY_value', 'use_client_certificate': True}, 'Boto': {'proxy': 'CLOUDSDK_PROXY_ADDRESS_value', 'proxy_type': 'CLOUDSDK_PROXY_TYPE_value', 'proxy_port': 'CLOUDSDK_PROXY_PORT_value', 'proxy_user': 'CLOUDSDK_PROXY_USERNAME_value', 'proxy_pass': 'CLOUDSDK_PROXY_PASSWORD_value', 'proxy_rdns': 'CLOUDSDK_PROXY_RDNS_value', 'http_socket_timeout': 'HTTP_TIMEOUT_value', 'ca_certificates_file': 'CA_CERTS_FILE_value', 'https_validate_certificates': False, 'max_retry_delay': 'BASE_RETRY_DELAY_value', 'num_retries': 'MAX_RETRIES_value'}, 'GSUtil': {'check_hashes': 'CHECK_HASHES_value', 'default_project_id': 'CLOUDSDK_CORE_PROJECT_value', 'disable_analytics_prompt': 'USAGE_REPORTING_value', 'use_magicfile': 'USE_MAGICFILE_value', 'parallel_composite_upload_threshold': '100M', 'resumable_threshold': '256K', 'rsync_buffer_lines': '32000'}, 'OAuth2': {'client_id': 'CLOUDSDK_AUTH_CLIENT_ID_value', 'client_secret': 'AUTH_CLIENT_SECRET_value', 'provider_authorization_uri': 'CLOUDSDK_AUTH_AUTH_HOST_value', 'provider_token_uri': 'CLOUDSDK_AUTH_TOKEN_HOST_value'}}):
        flags, env_vars = self._fake_command._translate_boto_config()
        self.assertEqual(flags, [])
        self.maxDiff = None
        self.assertDictEqual(env_vars, {'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID_value', 'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY_value', 'CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE': True, 'CLOUDSDK_STORAGE_GS_XML_ACCESS_KEY_ID': 'CLOUDSDK_STORAGE_GS_XML_ACCESS_KEY_ID_value', 'CLOUDSDK_STORAGE_GS_XML_SECRET_ACCESS_KEY': 'CLOUDSDK_STORAGE_GS_XML_SECRET_ACCESS_KEY_value', 'CLOUDSDK_PROXY_ADDRESS': 'CLOUDSDK_PROXY_ADDRESS_value', 'CLOUDSDK_PROXY_ADDRESS': 'CLOUDSDK_PROXY_ADDRESS_value', 'CLOUDSDK_PROXY_TYPE': 'CLOUDSDK_PROXY_TYPE_value', 'CLOUDSDK_PROXY_PORT': 'CLOUDSDK_PROXY_PORT_value', 'CLOUDSDK_PROXY_USERNAME': 'CLOUDSDK_PROXY_USERNAME_value', 'CLOUDSDK_PROXY_PASSWORD': 'CLOUDSDK_PROXY_PASSWORD_value', 'CLOUDSDK_PROXY_RDNS': 'CLOUDSDK_PROXY_RDNS_value', 'CLOUDSDK_CORE_HTTP_TIMEOUT': 'HTTP_TIMEOUT_value', 'CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE': 'CA_CERTS_FILE_value', 'CLOUDSDK_AUTH_DISABLE_SSL_VALIDATION': True, 'CLOUDSDK_STORAGE_BASE_RETRY_DELAY': 'BASE_RETRY_DELAY_value', 'CLOUDSDK_STORAGE_MAX_RETRIES': 'MAX_RETRIES_value', 'CLOUDSDK_STORAGE_CHECK_HASHES': 'CHECK_HASHES_value', 'CLOUDSDK_CORE_PROJECT': 'CLOUDSDK_CORE_PROJECT_value', 'CLOUDSDK_CORE_DISABLE_USAGE_REPORTING': 'USAGE_REPORTING_value', 'CLOUDSDK_STORAGE_USE_MAGICFILE': 'USE_MAGICFILE_value', 'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD': '100M', 'CLOUDSDK_STORAGE_RESUMABLE_THRESHOLD': '256K', 'CLOUDSDK_STORAGE_RSYNC_LIST_CHUNK_SIZE': '32000', 'CLOUDSDK_AUTH_CLIENT_ID': 'CLOUDSDK_AUTH_CLIENT_ID_value', 'CLOUDSDK_AUTH_CLIENT_SECRET': 'AUTH_CLIENT_SECRET_value', 'CLOUDSDK_AUTH_AUTH_HOST': 'CLOUDSDK_AUTH_AUTH_HOST_value', 'CLOUDSDK_AUTH_TOKEN_HOST': 'CLOUDSDK_AUTH_TOKEN_HOST_value'})