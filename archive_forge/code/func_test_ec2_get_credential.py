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
def test_ec2_get_credential(self):
    ec2_cred = self._get_ec2_cred()
    uri = '/'.join([self._get_ec2_cred_uri(), ec2_cred['access']])
    r = self.get(uri)
    self.assertDictEqual(ec2_cred, r.result['credential'])
    self.assertThat(ec2_cred['links']['self'], matchers.EndsWith(uri))