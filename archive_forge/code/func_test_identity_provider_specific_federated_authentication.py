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
def test_identity_provider_specific_federated_authentication(self):
    environment = {self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]}
    environment.update(mapping_fixtures.EMPLOYEE_ASSERTION)
    with self.make_request(environ=environment, query_string='origin=%s' % self.ORIGIN):
        resp = auth_api.AuthFederationWebSSOIDPsResource._perform_auth(self.idp['id'], self.PROTOCOL)
    self.assertIn(self.TRUSTED_DASHBOARD.encode('utf-8'), resp.data)