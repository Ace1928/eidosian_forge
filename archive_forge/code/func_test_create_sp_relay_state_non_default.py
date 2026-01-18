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
def test_create_sp_relay_state_non_default(self):
    """Create an SP with custom relay state."""
    url = self.base_url(suffix=uuid.uuid4().hex)
    sp = core.new_service_provider_ref()
    non_default_prefix = uuid.uuid4().hex
    sp['relay_state_prefix'] = non_default_prefix
    resp = self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
    sp_result = resp.result['service_provider']
    self.assertEqual(non_default_prefix, sp_result['relay_state_prefix'])