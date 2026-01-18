import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_subprojects(self):
    parent_project = fixtures.Project(self.client, self.test_domain.id)
    self.useFixture(parent_project)
    child_project_one = fixtures.Project(self.client, self.test_domain.id, parent=parent_project.id)
    self.useFixture(child_project_one)
    child_project_two = fixtures.Project(self.client, self.test_domain.id, parent=parent_project.id)
    self.useFixture(child_project_two)
    projects = self.client.projects.list(parent=parent_project.id)
    for project in projects:
        self.check_project(project)
    self.assertIn(child_project_one.entity, projects)
    self.assertIn(child_project_two.entity, projects)
    self.assertNotIn(parent_project.entity, projects)