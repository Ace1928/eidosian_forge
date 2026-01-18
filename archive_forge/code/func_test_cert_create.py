import copy
import testtools
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import certificates
def test_cert_create(self):
    cert = self.mgr.create(**CREATE_CERT)
    expect = [('POST', '/v1/certificates', {}, CREATE_CERT)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CERT2['cluster_uuid'], cert.cluster_uuid)
    self.assertEqual(CERT2['pem'], cert.pem)
    self.assertEqual(CERT2['csr'], cert.csr)