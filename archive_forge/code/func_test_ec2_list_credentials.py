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
def test_ec2_list_credentials(self):
    """Test ec2 credential listing."""
    self._get_ec2_cred()
    uri = self._get_ec2_cred_uri()
    r = self.get(uri)
    cred_list = r.result['credentials']
    self.assertEqual(1, len(cred_list))
    self.assertThat(r.result['links']['self'], matchers.EndsWith(uri))
    non_ec2_cred = unit.new_credential_ref(user_id=self.user_id, project_id=self.project_id)
    non_ec2_cred['type'] = uuid.uuid4().hex
    PROVIDERS.credential_api.create_credential(non_ec2_cred['id'], non_ec2_cred)
    r = self.get(uri)
    cred_list_2 = r.result['credentials']
    self.assertEqual(1, len(cred_list_2))
    self.assertEqual(cred_list[0], cred_list_2[0])