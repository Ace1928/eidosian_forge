import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_list_delete_token_shows_in_event_list(self):
    self.role_data_fixtures()
    events = self.get('/OS-REVOKE/events').json_body['events']
    self.assertEqual([], events)
    scoped_token = self.get_scoped_token()
    headers = {'X-Subject-Token': scoped_token}
    auth_req = self.build_authentication_request(token=scoped_token)
    response = self.v3_create_token(auth_req)
    token2 = response.json_body['token']
    headers2 = {'X-Subject-Token': response.headers['X-Subject-Token']}
    response = self.v3_create_token(auth_req)
    response.json_body['token']
    headers3 = {'X-Subject-Token': response.headers['X-Subject-Token']}
    self.head('/auth/tokens', headers=headers, expected_status=http.client.OK)
    self.head('/auth/tokens', headers=headers2, expected_status=http.client.OK)
    self.head('/auth/tokens', headers=headers3, expected_status=http.client.OK)
    self.delete('/auth/tokens', headers=headers)
    events_response = self.get('/OS-REVOKE/events').json_body
    events = events_response['events']
    self.assertEqual(1, len(events))
    self.assertEventDataInList(events, audit_id=token2['audit_ids'][1])
    self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
    self.head('/auth/tokens', headers=headers2, expected_status=http.client.OK)
    self.head('/auth/tokens', headers=headers3, expected_status=http.client.OK)