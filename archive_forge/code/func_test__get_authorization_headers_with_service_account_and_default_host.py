import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
def test__get_authorization_headers_with_service_account_and_default_host(self):
    credentials = mock.create_autospec(service_account.Credentials)
    request = mock.create_autospec(transport.Request)
    default_host = 'pubsub.googleapis.com'
    plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request, default_host=default_host)
    context = mock.create_autospec(grpc.AuthMetadataContext, instance=True)
    context.method_name = 'methodName'
    context.service_url = 'https://pubsub.googleapis.com/methodName'
    plugin._get_authorization_headers(context)
    credentials._create_self_signed_jwt.assert_called_once_with('https://{}/'.format(default_host))