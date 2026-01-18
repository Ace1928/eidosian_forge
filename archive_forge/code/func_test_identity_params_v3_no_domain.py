from openstack import exceptions
from openstack.tests.unit import base
def test_identity_params_v3_no_domain(self):
    project_data = self._get_project_data(v3=True)
    self.assertRaises(exceptions.SDKException, self.cloud._get_identity_params, domain_id=None, project=project_data.project_name)
    self.assert_calls()