import uuid
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import auth
def test_get_projects(self):
    body = {'projects': [self.create_resource(), self.create_resource(), self.create_resource()]}
    self.stub_url('GET', ['auth', 'projects'], json=body)
    projects = self.client.auth.projects()
    self.assertEqual(3, len(projects))
    for p in projects:
        self.assertIsInstance(p, auth.Project)