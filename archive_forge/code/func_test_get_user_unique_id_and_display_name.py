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
def test_get_user_unique_id_and_display_name(self):
    mapping = mapping_fixtures.MAPPING_USER_IDS
    assertion = mapping_fixtures.ADMIN_ASSERTION
    FAKE_MAPPING_ID = uuid.uuid4().hex
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)
    with self.flask_app.test_request_context(environ_base={'REMOTE_USER': 'remote_user'}):
        resource_api_mock = mock.patch('keystone.resource.core.DomainConfigManager')
        idp_domain_id = uuid.uuid4().hex
        mapped.validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api_mock)
    self.assertEqual('remote_user', mapped_properties['user']['name'])
    self.assertEqual('bob', mapped_properties['user']['id'])
    self.assertEqual(idp_domain_id, mapped_properties['user']['domain']['id'])