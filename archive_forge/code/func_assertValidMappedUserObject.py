import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def assertValidMappedUserObject(self, mapped_properties, user_type='ephemeral', domain_id=None):
    """Check whether mapped properties object has 'user' within.

        According to today's rules, RuleProcessor does not have to issue user's
        id or name. What's actually required is user's type.
        """
    self.assertIn('user', mapped_properties, message='Missing user object in mapped properties')
    user = mapped_properties['user']
    self.assertIn('type', user)
    self.assertEqual(user_type, user['type'])
    if domain_id:
        domain = user['domain']
        domain_name_or_id = domain.get('id') or domain.get('name')
        self.assertEqual(domain_id, domain_name_or_id)