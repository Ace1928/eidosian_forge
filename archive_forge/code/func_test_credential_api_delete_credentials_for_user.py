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
def test_credential_api_delete_credentials_for_user(self):
    PROVIDERS.credential_api.delete_credentials_for_user(self.user_id)
    self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, credential_id=self.credential['id'])