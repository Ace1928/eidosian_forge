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
@pytest.fixture(params=[dns_access_direct, dns_access_client_library])
def dns_access(request, http_request, service_account_info):

    def wrapper():
        return request.param(http_request, service_account_info['project_id'])
    yield wrapper