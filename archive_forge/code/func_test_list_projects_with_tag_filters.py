import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_projects_with_tag_filters(self):
    project_one = fixtures.Project(self.client, self.test_domain.id, tags=['tag1'])
    project_two = fixtures.Project(self.client, self.test_domain.id, tags=['tag1', 'tag2'])
    project_three = fixtures.Project(self.client, self.test_domain.id, tags=['tag2', 'tag3'])
    self.useFixture(project_one)
    self.useFixture(project_two)
    self.useFixture(project_three)
    projects = self.client.projects.list(tags='tag1')
    project_ids = []
    for project in projects:
        project_ids.append(project.id)
    self.assertIn(project_one.id, project_ids)
    projects = self.client.projects.list(tags_any='tag1')
    project_ids = []
    for project in projects:
        project_ids.append(project.id)
    self.assertIn(project_one.id, project_ids)
    self.assertIn(project_two.id, project_ids)
    projects = self.client.projects.list(not_tags='tag1')
    project_ids = []
    for project in projects:
        project_ids.append(project.id)
    self.assertNotIn(project_one.id, project_ids)
    projects = self.client.projects.list(not_tags_any='tag1,tag2')
    project_ids = []
    for project in projects:
        project_ids.append(project.id)
    self.assertNotIn(project_one.id, project_ids)
    self.assertNotIn(project_two.id, project_ids)
    self.assertNotIn(project_three.id, project_ids)
    projects = self.client.projects.list(tags='tag1,tag2')
    project_ids = []
    for project in projects:
        project_ids.append(project.id)
    self.assertNotIn(project_one.id, project_ids)
    self.assertIn(project_two.id, project_ids)
    self.assertNotIn(project_three.id, project_ids)