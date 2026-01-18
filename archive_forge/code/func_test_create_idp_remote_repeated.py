import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_create_idp_remote_repeated(self):
    """Create two IdentityProvider entities with some remote_ids.

        A remote_id is the same for both so the second IdP is not
        created because of the uniqueness of the remote_ids

        Expect HTTP 409 Conflict code for the latter call.

        """
    body = self.default_body.copy()
    repeated_remote_id = uuid.uuid4().hex
    body['remote_ids'] = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, repeated_remote_id]
    self._create_default_idp(body=body)
    url = self.base_url(suffix=uuid.uuid4().hex)
    body['remote_ids'] = [uuid.uuid4().hex, repeated_remote_id]
    resp = self.put(url, body={'identity_provider': body}, expected_status=http.client.CONFLICT)
    resp_data = jsonutils.loads(resp.body)
    self.assertIn('Duplicate remote ID', resp_data.get('error', {}).get('message'))