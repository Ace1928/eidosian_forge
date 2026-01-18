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
def test_crud_protocol_without_protocol_id_in_url(self):
    idp_id, _ = self._create_and_decapsulate_response()
    mapping_id = uuid.uuid4().hex
    self._create_mapping(mapping_id=mapping_id)
    protocol = {'id': uuid.uuid4().hex, 'mapping_id': mapping_id}
    with self.test_client() as c:
        token = self.get_scoped_token()
        c.delete('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols' % {'idp_id': idp_id}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.patch('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.put('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.delete('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.patch('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
        c.put('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)