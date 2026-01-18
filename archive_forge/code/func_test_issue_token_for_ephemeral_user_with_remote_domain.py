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
def test_issue_token_for_ephemeral_user_with_remote_domain(self):
    """Test ephemeral user is created in the domain set by assertion.

        Shadow user may belong to the domain set by the assertion data.
        To verify that:
         - precreate domain used later in the assertion
         - update mapping to unclude user domain name coming from assertion
         - auth user
         - verify user domain is not the IDP domain

        """
    domain_ref = unit.new_domain_ref(name='user_domain')
    PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    PROVIDERS.federation_api.update_mapping(self.mapping['id'], mapping_fixtures.MAPPING_EPHEMERAL_USER_REMOTE_DOMAIN)
    r = self._issue_unscoped_token(assertion='USER_WITH_DOMAIN_ASSERTION')
    self.assertEqual(r.user_domain['id'], domain_ref['id'])
    self.assertNotEqual(r.user_domain['id'], self.idp['domain_id'])