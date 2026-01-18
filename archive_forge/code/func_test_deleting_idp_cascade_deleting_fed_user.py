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
def test_deleting_idp_cascade_deleting_fed_user(self):
    token = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN)
    federated_info = token.json_body['token']['user']['OS-FEDERATION']
    idp_id = federated_info['identity_provider']['id']
    hints = driver_hints.Hints()
    hints.add_filter('idp_id', idp_id)
    fed_users = PROVIDERS.shadow_users_api.get_federated_users(hints)
    self.assertEqual(3, len(fed_users))
    idp_domain_id = PROVIDERS.federation_api.get_idp(idp_id)['domain_id']
    for fed_user in fed_users:
        self.assertEqual(idp_domain_id, fed_user['domain_id'])
    PROVIDERS.federation_api.delete_idp(idp_id)
    hints = driver_hints.Hints()
    hints.add_filter('idp_id', idp_id)
    fed_users = PROVIDERS.shadow_users_api.get_federated_users(hints)
    self.assertEqual([], fed_users)