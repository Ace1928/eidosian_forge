import copy
import testtools
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import certificates
def test_cert_show_by_id(self):
    cert = self.mgr.get(CERT1['cluster_uuid'])
    expect = [('GET', '/v1/certificates/%s' % CERT1['cluster_uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CERT1['cluster_uuid'], cert.cluster_uuid)
    self.assertEqual(CERT1['pem'], cert.pem)