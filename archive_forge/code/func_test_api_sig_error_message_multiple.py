import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_api_sig_error_message_multiple(self):
    title = 'this error is the first error!'
    detail = 'it is a totally made up error'
    error_message = {'errors': [{'request_id': uuid.uuid4().hex, 'code': 'phoney.bologna.error', 'status': 500, 'title': title, 'detail': detail, 'links': [{'rel': 'help', 'href': 'https://openstack.org'}]}, {'request_id': uuid.uuid4().hex, 'code': 'phoney.bologna.error', 'status': 500, 'title': 'some other error', 'detail': detail, 'links': [{'rel': 'help', 'href': 'https://openstack.org'}]}]}
    payload = json.dumps(error_message)
    self.stub_url('GET', status_code=9000, text=payload, headers={'Content-Type': 'application/json'})
    session = client_session.Session()
    msg = 'Multiple error responses, showing first only: {} (HTTP 9000)'.format(title)
    try:
        session.get(self.TEST_URL)
    except exceptions.HttpError as ex:
        self.assertEqual(ex.message, msg)
        self.assertEqual(ex.details, detail)