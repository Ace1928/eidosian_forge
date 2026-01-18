import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_add_invalid_tags(self):
    project_one = fixtures.Project(self.client, self.test_domain.id)
    self.useFixture(project_one)
    self.assertRaises(exceptions.BadRequest, project_one.add_tag, ',')
    self.assertRaises(exceptions.BadRequest, project_one.add_tag, '/')
    self.assertRaises(exceptions.BadRequest, project_one.add_tag, '')