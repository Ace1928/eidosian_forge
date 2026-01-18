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
def test_list_head_service_providers(self):
    """Test listing of service provider objects.

        Add two new service providers. List all available service providers.
        Expect to get list of three service providers (one created by setUp())
        Test if attributes match.

        """
    ref_service_providers = {uuid.uuid4().hex: core.new_service_provider_ref(), uuid.uuid4().hex: core.new_service_provider_ref()}
    for id, sp in ref_service_providers.items():
        url = self.base_url(suffix=id)
        self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
    ref_service_providers[self.SERVICE_PROVIDER_ID] = self.SP_REF
    for id, sp in ref_service_providers.items():
        sp['id'] = id
    url = self.base_url()
    resp = self.get(url)
    service_providers = resp.result
    for service_provider in service_providers['service_providers']:
        id = service_provider['id']
        self.assertValidEntity(service_provider, ref=ref_service_providers[id], keys_to_check=self.SP_KEYS)
    self.head(url, expected_status=http.client.OK)