import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_endpoint_with_version_hostname(self):
    gc = client.Client(2, 'http://v1.example.com')
    self.assertEqual('http://v1.example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v2.client.Client)