import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
def test_get_ca_certificate(self):
    self.stub_url('GET', ['certificates', 'ca'], headers={'Content-Type': 'text/html; charset=UTF-8'}, text=self.examples.SIGNING_CA)
    res = self.client.certificates.get_ca_certificate()
    self.assertEqual(self.examples.SIGNING_CA, res)