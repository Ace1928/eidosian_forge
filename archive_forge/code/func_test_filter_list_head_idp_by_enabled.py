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
def test_filter_list_head_idp_by_enabled(self):

    def get_id(resp):
        r = self._fetch_attribute_from_response(resp, 'identity_provider')
        return r.get('id')
    idp1_id = get_id(self._create_default_idp())
    body = self.default_body.copy()
    body['enabled'] = False
    idp2_id = get_id(self._create_default_idp(body=body))
    url = self.base_url()
    resp = self.get(url)
    entities = self._fetch_attribute_from_response(resp, 'identity_providers')
    entities_ids = [e['id'] for e in entities]
    self.assertCountEqual(entities_ids, [idp1_id, idp2_id])
    url = self.base_url() + '?enabled=True'
    resp = self.get(url)
    filtered_service_list = resp.json['identity_providers']
    self.assertThat(filtered_service_list, matchers.HasLength(1))
    self.assertEqual(idp1_id, filtered_service_list[0].get('id'))
    self.head(url, expected_status=http.client.OK)