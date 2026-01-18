from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_json_request(self):
    session = utils.mockSession({}, status_code=200)
    req_id = 'req-7b081d28-8272-45f4-9cf6-89649c1c7a1a'
    client = _session_client(session=session, additional_headers={'foo': 'bar'}, global_request_id=req_id)
    client.json_request('GET', 'url')
    session.request.assert_called_once_with('url', 'GET', raise_exc=False, auth=None, headers={'foo': 'bar', 'X-OpenStack-Request-ID': req_id, 'Content-Type': 'application/json', 'Accept': 'application/json', 'X-OpenStack-Ironic-API-Version': '1.6'}, endpoint_filter={'interface': 'publicURL', 'service_type': 'baremetal', 'region_name': ''}, endpoint_override='http://localhost:1234', user_agent=http.USER_AGENT)