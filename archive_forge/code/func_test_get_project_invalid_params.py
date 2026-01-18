import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_project_invalid_params(self):
    self.assertRaises(exceptions.ValidationError, self.client.projects.get, self.test_project.id, subtree_as_list=True, subtree_as_ids=True)
    self.assertRaises(exceptions.ValidationError, self.client.projects.get, self.test_project.id, parents_as_list=True, parents_as_ids=True)