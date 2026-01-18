import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
def test_get_signing_certificate(self):
    self.stub_url('GET', ['certificates', 'signing'], headers={'Content-Type': 'text/html; charset=UTF-8'}, text=self.examples.SIGNING_CERT)
    res = self.client.certificates.get_signing_certificate()
    self.assertEqual(self.examples.SIGNING_CERT, res)