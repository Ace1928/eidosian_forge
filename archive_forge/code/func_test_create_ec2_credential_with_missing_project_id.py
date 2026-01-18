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
def test_create_ec2_credential_with_missing_project_id(self):
    """Test Creating ec2 credential with missing project_id.

        Call ``POST /credentials``.
        """
    _, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=None)
    self.post('/credentials', body={'credential': ref}, expected_status=http.client.BAD_REQUEST)