from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
def test_duplicate_method(self):
    self.useFixture(auth_plugins.ConfigAuthPlugins(self.config_fixture, ['external', 'external']))
    auth.core.load_auth_methods()
    self.assertIn('external', auth.core.AUTH_METHODS)