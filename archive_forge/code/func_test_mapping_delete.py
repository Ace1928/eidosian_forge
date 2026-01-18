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
def test_mapping_delete(self):
    url = self.MAPPING_URL + '%(mapping_id)s'
    resp = self._create_default_mapping_entry()
    mapping_id = self._get_id_from_response(resp)
    url = url % {'mapping_id': str(mapping_id)}
    resp = self.delete(url)
    self.assertResponseStatus(resp, http.client.NO_CONTENT)
    self.get(url, expected_status=http.client.NOT_FOUND)