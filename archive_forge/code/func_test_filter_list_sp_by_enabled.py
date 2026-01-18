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
def test_filter_list_sp_by_enabled(self):

    def get_id(resp):
        sp = resp.result.get('service_provider')
        return sp.get('id')
    sp1_id = get_id(self._create_default_sp())
    sp2_ref = core.new_service_provider_ref()
    sp2_ref['enabled'] = False
    sp2_id = get_id(self._create_default_sp(body=sp2_ref))
    url = self.base_url()
    resp = self.get(url)
    sps = resp.result.get('service_providers')
    entities_ids = [e['id'] for e in sps]
    self.assertIn(sp1_id, entities_ids)
    self.assertIn(sp2_id, entities_ids)
    url = self.base_url() + '?enabled=True'
    resp = self.get(url)
    sps = resp.result.get('service_providers')
    entities_ids = [e['id'] for e in sps]
    self.assertIn(sp1_id, entities_ids)
    self.assertNotIn(sp2_id, entities_ids)