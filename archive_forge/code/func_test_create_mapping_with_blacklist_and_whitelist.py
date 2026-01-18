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
def test_create_mapping_with_blacklist_and_whitelist(self):
    """Test for adding whitelist and blacklist in the rule.

        Server should respond with HTTP 400 Bad Request error upon discovering
        both ``whitelist`` and ``blacklist`` keywords in the same rule.

        """
    url = self.MAPPING_URL + uuid.uuid4().hex
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_AND_BLACKLIST
    self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping})