import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def test_missing_oauth_headers(self):
    endpoint = '/OS-OAUTH1/request_token'
    client = oauth1.Client(uuid.uuid4().hex, client_secret=uuid.uuid4().hex, signature_method=oauth1.SIG_HMAC, callback_uri='oob')
    headers = {'requested_project_id': uuid.uuid4().hex}
    _url, headers, _body = client.sign(self.base_url + endpoint, http_method='POST', headers=headers)
    del headers['Authorization']
    self.post(endpoint, headers=headers, expected_status=http.client.INTERNAL_SERVER_ERROR)