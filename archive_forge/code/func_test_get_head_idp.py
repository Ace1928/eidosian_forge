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
def test_get_head_idp(self):
    """Create and later fetch IdP."""
    body = self._http_idp_input()
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    body['domain_id'] = domain['id']
    default_resp = self._create_default_idp(body=body)
    default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
    idp_id = default_idp.get('id')
    url = self.base_url(suffix=idp_id)
    resp = self.get(url)
    body_keys = list(body)
    self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=body_keys, ref=body)
    self.head(url, expected_status=http.client.OK)