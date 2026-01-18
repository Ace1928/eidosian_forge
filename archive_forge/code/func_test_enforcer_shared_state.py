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
def test_enforcer_shared_state(self):
    enforcer = rbac_enforcer.enforcer.RBACEnforcer()
    enforcer2 = rbac_enforcer.enforcer.RBACEnforcer()
    self.assertIsNotNone(enforcer._enforcer)
    self.assertEqual(enforcer._enforcer, enforcer2._enforcer)
    setattr(enforcer, '_test_attr', uuid.uuid4().hex)
    self.assertEqual(enforcer._test_attr, enforcer2._test_attr)