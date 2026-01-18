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
def test_issue_the_same_unscoped_token_with_user_deleted(self):
    r = self._issue_unscoped_token()
    token = render_token.render_token_response_from_model(r)['token']
    user1 = token['user']
    user_id1 = user1.pop('id')
    PROVIDERS.identity_api.delete_user(user_id1)
    r = self._issue_unscoped_token()
    token = render_token.render_token_response_from_model(r)['token']
    user2 = token['user']
    user_id2 = user2.pop('id')
    self.assertIsNot(user_id2, user_id1)
    self.assertEqual(user1, user2)