import fixtures
import testresources
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib import simple_cert
def test_get_certificates(self):
    self.stub_url('GET', ['OS-SIMPLE-CERT', 'certificates'], headers={'Content-Type': 'application/x-pem-file'}, text=self.examples.SIGNING_CERT)
    res = self.client.simple_cert.get_certificates()
    self.assertEqual(self.examples.SIGNING_CERT, res)