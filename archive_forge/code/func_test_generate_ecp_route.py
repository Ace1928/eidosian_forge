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
def test_generate_ecp_route(self):
    """Test that the ECP generation endpoint produces XML.

        The ECP endpoint /v3/auth/OS-FEDERATION/saml2/ecp should take the same
        input as the SAML generation endpoint (scoped token ID + Service
        Provider ID).
        The controller should return a SAML assertion that is wrapped in a
        SOAP envelope.
        """
    self.config_fixture.config(group='saml', idp_entity_id=self.ISSUER)
    token_id = self._fetch_valid_token()
    body = self._create_generate_saml_request(token_id, self.SERVICE_PROVDIER_ID)
    with mock.patch.object(keystone_idp, '_sign_assertion', return_value=self.signed_assertion):
        http_response = self.post(self.ECP_GENERATION_ROUTE, body=body, response_content_type='text/xml', expected_status=http.client.OK)
    env_response = etree.fromstring(http_response.result)
    header = env_response[0]
    prefix = CONF.saml.relay_state_prefix
    self.assertThat(header[0].text, matchers.StartsWith(prefix))
    body = env_response[1]
    response = body[0]
    issuer = response[0]
    assertion = response[2]
    self.assertEqual(self.RECIPIENT, response.get('Destination'))
    self.assertEqual(self.ISSUER, issuer.text)
    user_attribute = assertion[4][0]
    self.assertIsInstance(user_attribute[0].text, str)
    user_domain_attribute = assertion[4][1]
    self.assertIsInstance(user_domain_attribute[0].text, str)
    role_attribute = assertion[4][2]
    self.assertIsInstance(role_attribute[0].text, str)
    project_attribute = assertion[4][3]
    self.assertIsInstance(project_attribute[0].text, str)
    project_domain_attribute = assertion[4][4]
    self.assertIsInstance(project_domain_attribute[0].text, str)
    group_attribute = assertion[4][5]
    self.assertIsInstance(group_attribute[0].text, str)