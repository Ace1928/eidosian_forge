import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_project_invalid_tags(self):
    project_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.test_domain.id, 'enabled': True, 'description': uuid.uuid4().hex, 'tags': ','}
    self.assertRaises(exceptions.BadRequest, self.client.projects.create, **project_ref)
    project_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.test_domain.id, 'enabled': True, 'description': uuid.uuid4().hex, 'tags': '/'}
    self.assertRaises(exceptions.BadRequest, self.client.projects.create, **project_ref)
    project_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.test_domain.id, 'enabled': True, 'description': uuid.uuid4().hex, 'tags': ''}
    self.assertRaises(exceptions.BadRequest, self.client.projects.create, **project_ref)