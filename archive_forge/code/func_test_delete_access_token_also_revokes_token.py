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
def test_delete_access_token_also_revokes_token(self):
    self.test_oauth_flow()
    access_token_key = self.access_token.key.decode()
    resp = self.delete('/users/%(user)s/OS-OAUTH1/access_tokens/%(auth)s' % {'user': self.user_id, 'auth': access_token_key})
    self.assertResponseStatus(resp, http.client.NO_CONTENT)
    headers = {'X-Subject-Token': self.keystone_token_id, 'X-Auth-Token': self.keystone_token_id}
    self.get('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)