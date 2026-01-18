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
def test_bad_authorizing_roles_id(self):
    consumer = self._create_single_consumer()
    consumer_id = consumer['id']
    consumer_secret = consumer['secret']
    consumer = {'key': consumer_id, 'secret': consumer_secret}
    new_role = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
    PROVIDERS.role_api.create_role(new_role['id'], new_role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=self.user_id, project_id=self.project_id, role_id=new_role['id'])
    url, headers = self._create_request_token(consumer, self.project_id)
    content = self.post(url, headers=headers, response_content_type='application/x-www-form-urlencoded')
    credentials = _urllib_parse_qs_text_keys(content.result)
    request_key = credentials['oauth_token'][0]
    PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user_id, self.project_id, new_role['id'])
    url = self._authorize_request_token(request_key)
    body = {'roles': [{'id': new_role['id']}]}
    self.put(path=url, body=body, expected_status=http.client.UNAUTHORIZED)