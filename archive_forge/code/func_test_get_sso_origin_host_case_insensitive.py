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
def test_get_sso_origin_host_case_insensitive(self):
    environ = {'QUERY_STRING': 'origin=http://horizon.com'}
    with self.make_request(environ=environ):
        host = auth_api._get_sso_origin_host()
        self.assertEqual('http://horizon.com', host)
        self.config_fixture.config(group='federation', trusted_dashboard=['http://Horizon.com'])
        host = auth_api._get_sso_origin_host()
        self.assertEqual('http://horizon.com', host)