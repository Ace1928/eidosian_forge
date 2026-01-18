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
def test_keystone_token_is_valid(self):
    self.test_oauth_flow()
    headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
    r = self.get('/auth/tokens', headers=headers)
    self.assertValidTokenResponse(r, self.user)
    oauth_section = r.result['token']['OS-OAUTH1']
    self.assertEqual(self.access_token.key.decode(), oauth_section['access_token_id'])
    self.assertEqual(self.consumer['key'], oauth_section['consumer_id'])
    roles_list = r.result['token']['roles']
    self.assertEqual(self.role_id, roles_list[0]['id'])
    ref = unit.new_user_ref(domain_id=self.domain_id)
    r = self.admin_request(path='/v3/users', headers=headers, method='POST', body={'user': ref})
    self.assertValidUserResponse(r, ref)