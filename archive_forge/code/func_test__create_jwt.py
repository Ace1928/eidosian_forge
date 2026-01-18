import copy
import datetime
import json
import os
import mock
import pytest  # type: ignore
import requests
import six
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
from google.oauth2 import gdch_credentials
from google.oauth2.gdch_credentials import ServiceAccountCredentials
def test__create_jwt(self):
    creds = ServiceAccountCredentials.from_service_account_file(self.JSON_PATH)
    with mock.patch('google.auth._helpers.utcnow') as utcnow:
        utcnow.return_value = datetime.datetime.now()
        jwt_token = creds._create_jwt()
        header, payload, _, _ = jwt._unverified_decode(jwt_token)
    expected_iss_sub_value = 'system:serviceaccount:project_foo:service_identity_name'
    assert isinstance(jwt_token, six.text_type)
    assert header['alg'] == 'ES256'
    assert header['kid'] == self.PRIVATE_KEY_ID
    assert payload['iss'] == expected_iss_sub_value
    assert payload['sub'] == expected_iss_sub_value
    assert payload['aud'] == self.AUDIENCE
    assert payload['exp'] == payload['iat'] + 3600