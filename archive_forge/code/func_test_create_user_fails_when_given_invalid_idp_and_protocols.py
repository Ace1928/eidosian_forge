import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_create_user_fails_when_given_invalid_idp_and_protocols(self):
    """Call ``POST /users`` with invalid idp and protocol to fail."""
    idp, protocol = self._create_federated_attributes()
    ref = unit.new_user_ref(domain_id=self.domain_id)
    ref['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'fakeprotocol_id', 'unique_id': uuid.uuid4().hex}]}]
    self.post('/users', body={'user': ref}, token=self.get_admin_token(), expected_status=http.client.BAD_REQUEST)
    ref['federated'][0]['idp_id'] = idp['id']
    self.post('/users', body={'user': ref}, token=self.get_admin_token(), expected_status=http.client.BAD_REQUEST)