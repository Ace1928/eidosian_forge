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
def test_list_head_protocols(self):
    """Create set of protocols and later list them.

        Compare input and output id sets.

        """
    resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
    iterations = random.randint(0, 16)
    protocol_ids = []
    for _ in range(iterations):
        resp, _, proto = self._assign_protocol_to_idp(idp_id=idp_id, expected_status=http.client.CREATED)
        proto_id = self._fetch_attribute_from_response(resp, 'protocol')
        proto_id = proto_id['id']
        protocol_ids.append(proto_id)
    url = '%s/protocols' % idp_id
    url = self.base_url(suffix=url)
    resp = self.get(url)
    self.assertValidListResponse(resp, 'protocols', dummy_validator, keys_to_check=['id'])
    entities = self._fetch_attribute_from_response(resp, 'protocols')
    entities = set([entity['id'] for entity in entities])
    protocols_intersection = entities.intersection(protocol_ids)
    self.assertEqual(protocols_intersection, set(protocol_ids))
    self.head(url, expected_status=http.client.OK)