from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_list_project_with_project_domain(self):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    self.ep_filter_mock.list_endpoints_for_project.return_value = [self.endpoint]
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.name, '--project-domain', domain.name]
    verifylist = [('project', project.name), ('project_domain', domain.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.ep_filter_mock.list_endpoints_for_project.assert_called_with(project=project.id)
    self.assertEqual(self.columns, columns)
    datalist = ((self.endpoint.id, self.endpoint.region, self.service.name, self.service.type, True, self.endpoint.interface, self.endpoint.url),)
    self.assertEqual(datalist, tuple(data))