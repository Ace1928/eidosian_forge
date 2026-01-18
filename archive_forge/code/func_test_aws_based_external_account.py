import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
def test_aws_based_external_account(aws_oidc_credentials, service_account_info, dns_access, http_request):
    response = http_request(url='https://sts.amazonaws.com/?Action=AssumeRoleWithWebIdentity&Version=2011-06-15&DurationSeconds=3600&RoleSessionName=python-test&RoleArn={}&WebIdentityToken={}'.format(_ROLE_AWS, aws_oidc_credentials))
    assert response.status == 200
    data = response.data.decode('utf-8')
    with patch.dict(os.environ, {'AWS_REGION': 'us-east-2', 'AWS_ACCESS_KEY_ID': get_xml_value_by_tagname(data, 'AccessKeyId'), 'AWS_SECRET_ACCESS_KEY': get_xml_value_by_tagname(data, 'SecretAccessKey'), 'AWS_SESSION_TOKEN': get_xml_value_by_tagname(data, 'SessionToken')}):
        assert get_project_dns(dns_access, {'type': 'external_account', 'audience': _AUDIENCE_AWS, 'subject_token_type': 'urn:ietf:params:aws:token-type:aws4_request', 'token_url': 'https://sts.googleapis.com/v1/token', 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(service_account_info['client_email']), 'credential_source': {'environment_id': 'aws1', 'regional_cred_verification_url': 'https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15'}})