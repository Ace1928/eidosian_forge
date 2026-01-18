import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test_get_public_url(self):
    test_public_url = 'test_public_url'
    scObj = auth.ServiceCatalog()
    scObj.public_url = test_public_url
    self.assertEqual(test_public_url, scObj.get_public_url())