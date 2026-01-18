import uuid
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import auth
def test_get_domains(self):
    body = {'domains': [self.create_resource(), self.create_resource(), self.create_resource()]}
    self.stub_url('GET', ['auth', 'domains'], json=body)
    domains = self.client.auth.domains()
    self.assertEqual(3, len(domains))
    for d in domains:
        self.assertIsInstance(d, auth.Domain)