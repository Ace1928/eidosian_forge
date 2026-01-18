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
def test_local_user_local_domain(self):
    """Test that local users can have non-service domains assigned."""
    mapping = mapping_fixtures.MAPPING_LOCAL_USER_LOCAL_DOMAIN
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.CONTRACTOR_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties, user_type='local', domain_id=mapping_fixtures.LOCAL_DOMAIN)