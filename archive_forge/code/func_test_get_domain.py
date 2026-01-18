import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_get_domain(self):
    domain_data = self._get_domain_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[domain_data.domain_id]), status_code=200, json=domain_data.json_response)])
    domain = self.cloud.get_domain(domain_id=domain_data.domain_id)
    self.assertThat(domain.id, matchers.Equals(domain_data.domain_id))
    self.assertThat(domain.name, matchers.Equals(domain_data.domain_name))
    self.assert_calls()