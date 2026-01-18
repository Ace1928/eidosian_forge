import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_no_endpoint_error(self):
    self.assertRaises(ValueError, client.Client, None)