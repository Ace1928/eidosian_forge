import uuid
from keystoneclient.tests.unit.v3 import utils
def test_add_endpoint_to_project_via_obj(self):
    project_ref = self.new_project_ref()
    endpoint_ref = self.new_endpoint_ref()
    project = self.client.projects.resource_class(self.client.projects, project_ref, loaded=True)
    endpoint = self.client.endpoints.resource_class(self.client.endpoints, endpoint_ref, loaded=True)
    self.stub_url('PUT', [self.manager.OS_EP_FILTER_EXT, 'projects', project_ref['id'], 'endpoints', endpoint_ref['id']], status_code=201)
    self.manager.add_endpoint_to_project(project=project, endpoint=endpoint)