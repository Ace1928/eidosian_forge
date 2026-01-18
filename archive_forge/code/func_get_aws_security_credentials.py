import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
@_helpers.copy_docstring(AwsSecurityCredentialsSupplier)
def get_aws_security_credentials(self, context, request):
    env_aws_access_key_id = os.environ.get(environment_vars.AWS_ACCESS_KEY_ID)
    env_aws_secret_access_key = os.environ.get(environment_vars.AWS_SECRET_ACCESS_KEY)
    env_aws_session_token = os.environ.get(environment_vars.AWS_SESSION_TOKEN)
    if env_aws_access_key_id and env_aws_secret_access_key:
        return AwsSecurityCredentials(env_aws_access_key_id, env_aws_secret_access_key, env_aws_session_token)
    imdsv2_session_token = self._get_imdsv2_session_token(request)
    role_name = self._get_metadata_role_name(request, imdsv2_session_token)
    credentials = self._get_metadata_security_credentials(request, role_name, imdsv2_session_token)
    return AwsSecurityCredentials(credentials.get('AccessKeyId'), credentials.get('SecretAccessKey'), credentials.get('Token'))