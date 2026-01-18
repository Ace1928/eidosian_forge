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
def get_aws_region(self, context, request):
    env_aws_region = os.environ.get(environment_vars.AWS_REGION)
    if env_aws_region is not None:
        return env_aws_region
    env_aws_region = os.environ.get(environment_vars.AWS_DEFAULT_REGION)
    if env_aws_region is not None:
        return env_aws_region
    if not self._region_url:
        raise exceptions.RefreshError('Unable to determine AWS region')
    headers = None
    imdsv2_session_token = self._get_imdsv2_session_token(request)
    if imdsv2_session_token is not None:
        headers = {'X-aws-ec2-metadata-token': imdsv2_session_token}
    response = request(url=self._region_url, method='GET', headers=headers)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    if response.status != http_client.OK:
        raise exceptions.RefreshError('Unable to retrieve AWS region: {}'.format(response_body))
    return response_body[:-1]