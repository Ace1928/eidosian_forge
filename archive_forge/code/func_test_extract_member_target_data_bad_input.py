from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def test_extract_member_target_data_bad_input(self):
    self.assertEqual({}, self.enforcer._extract_member_target_data(member_target=None, member_target_type=uuid.uuid4().hex))
    self.assertEqual({}, self.enforcer._extract_member_target_data(member_target={}, member_target_type=None))