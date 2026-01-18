import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_list_credentials_filtered_by_user_id(self):
    """Call ``GET  /credentials?user_id={user_id}``."""
    credential = unit.new_credential_ref(user_id=uuid.uuid4().hex)
    PROVIDERS.credential_api.create_credential(credential['id'], credential)
    r = self.get('/credentials?user_id=%s' % self.user['id'])
    self.assertValidCredentialListResponse(r, ref=self.credential)
    for cred in r.result['credentials']:
        self.assertEqual(self.user['id'], cred['user_id'])