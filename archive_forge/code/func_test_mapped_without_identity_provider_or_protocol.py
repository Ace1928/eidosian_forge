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
@mock.patch('keystone.auth.plugins.mapped.PROVIDERS')
def test_mapped_without_identity_provider_or_protocol(self, mock_providers):
    mock_providers.resource_api = mock.Mock()
    mock_providers.federation_api = mock.Mock()
    mock_providers.identity_api = mock.Mock()
    mock_providers.assignment_api = mock.Mock()
    mock_providers.role_api = mock.Mock()
    test_mapped = mapped.Mapped()
    auth_payload = {'identity_provider': 'test_provider'}
    with self.make_request():
        self.assertRaises(exception.ValidationError, test_mapped.authenticate, auth_payload)
    auth_payload = {'protocol': 'saml2'}
    with self.make_request():
        self.assertRaises(exception.ValidationError, test_mapped.authenticate, auth_payload)