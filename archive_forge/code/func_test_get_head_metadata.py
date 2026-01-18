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
def test_get_head_metadata(self):
    self.config_fixture.config(group='saml', idp_metadata_path=XMLDIR + '/idp_saml2_metadata.xml')
    self.head(self.METADATA_URL, expected_status=http.client.OK)
    r = self.get(self.METADATA_URL, response_content_type='text/xml')
    self.assertEqual('text/xml', r.headers.get('Content-Type'))
    reference_file = _load_xml('idp_saml2_metadata.xml')
    reference_file = str.encode(reference_file)
    self.assertEqual(reference_file, r.result)