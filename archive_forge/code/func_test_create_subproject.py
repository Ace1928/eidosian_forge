import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_subproject(self):
    project_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.test_domain.id, 'enabled': True, 'description': uuid.uuid4().hex, 'parent': self.test_project.id}
    project = self.client.projects.create(**project_ref)
    self.addCleanup(self.client.projects.delete, project)
    self.check_project(project, project_ref)