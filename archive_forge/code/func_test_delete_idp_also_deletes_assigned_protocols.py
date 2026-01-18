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
def test_delete_idp_also_deletes_assigned_protocols(self):
    """Deleting an IdP will delete its assigned protocol."""
    default_resp = self._create_default_idp()
    default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
    idp_id = default_idp['id']
    protocol_id = uuid.uuid4().hex
    url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
    idp_url = self.base_url(suffix=idp_id)
    kwargs = {'expected_status': http.client.CREATED}
    resp, idp_id, proto = self._assign_protocol_to_idp(url=url, idp_id=idp_id, proto=protocol_id, **kwargs)
    self.assertEqual(1, len(PROVIDERS.federation_api.list_protocols(idp_id)))
    self.delete(idp_url)
    self.get(idp_url, expected_status=http.client.NOT_FOUND)
    self.assertEqual(0, len(PROVIDERS.federation_api.list_protocols(idp_id)))