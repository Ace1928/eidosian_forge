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
def test_check_idp_sso(self):
    metadata = self.generator.generate_metadata()
    idpsso_descriptor = metadata.idpsso_descriptor
    self.assertIsNotNone(metadata.idpsso_descriptor)
    self.assertEqual(federation_fixtures.IDP_SSO_ENDPOINT, idpsso_descriptor.single_sign_on_service.location)
    self.assertIsNotNone(idpsso_descriptor.organization)
    organization = idpsso_descriptor.organization
    self.assertEqual(federation_fixtures.IDP_ORGANIZATION_DISPLAY_NAME, organization.organization_display_name.text)
    self.assertEqual(federation_fixtures.IDP_ORGANIZATION_NAME, organization.organization_name.text)
    self.assertEqual(federation_fixtures.IDP_ORGANIZATION_URL, organization.organization_url.text)
    self.assertIsNotNone(idpsso_descriptor.contact_person)
    contact_person = idpsso_descriptor.contact_person
    self.assertEqual(federation_fixtures.IDP_CONTACT_GIVEN_NAME, contact_person.given_name.text)
    self.assertEqual(federation_fixtures.IDP_CONTACT_SURNAME, contact_person.sur_name.text)
    self.assertEqual(federation_fixtures.IDP_CONTACT_EMAIL, contact_person.email_address.text)
    self.assertEqual(federation_fixtures.IDP_CONTACT_TELEPHONE_NUMBER, contact_person.telephone_number.text)
    self.assertEqual(federation_fixtures.IDP_CONTACT_TYPE, contact_person.contact_type)