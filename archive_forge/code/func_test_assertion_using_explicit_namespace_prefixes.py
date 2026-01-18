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
def test_assertion_using_explicit_namespace_prefixes(self):

    def mocked_subprocess_check_output(*popenargs, **kwargs):
        if popenargs[0] != ['/usr/bin/which', CONF.saml.xmlsec1_binary]:
            filename = popenargs[0][-1]
            with open(filename, 'r') as f:
                assertion_content = f.read()
            return assertion_content
    with mock.patch.object(subprocess, 'check_output', side_effect=mocked_subprocess_check_output):
        generator = keystone_idp.SAMLGenerator()
        response = generator.samlize_token(self.ISSUER, self.RECIPIENT, self.SUBJECT, self.SUBJECT_DOMAIN, self.ROLES, self.PROJECT, self.PROJECT_DOMAIN, self.GROUPS)
        assertion_xml = response.assertion.to_string()
        self.assertIn(b'<saml:Assertion', assertion_xml)
        self.assertIn(('xmlns:saml="' + saml2.NAMESPACE + '"').encode('utf-8'), assertion_xml)
        self.assertIn(('xmlns:xmldsig="' + xmldsig.NAMESPACE).encode('utf-8'), assertion_xml)