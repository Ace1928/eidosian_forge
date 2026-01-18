import json
import os
from google.auth import _helpers
import google.auth.transport.requests
import google.auth.transport.urllib3
import pytest
import requests
import urllib3
@pytest.fixture
def authenticated_request(request_type):
    """A transport.request object that takes credentials"""
    if request_type == 'urllib3':

        def wrapper(credentials):
            return google.auth.transport.urllib3.AuthorizedHttp(credentials, http=URLLIB3_HTTP).request
        yield wrapper
    elif request_type == 'requests':

        def wrapper(credentials):
            session = google.auth.transport.requests.AuthorizedSession(credentials)
            session.verify = False
            return google.auth.transport.requests.Request(session)
        yield wrapper