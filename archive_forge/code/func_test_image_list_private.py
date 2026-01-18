from keystoneauth1 import session
from requests_mock.contrib import fixture
from openstackclient.api import image_v2
from openstackclient.tests.unit import utils
def test_image_list_private(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/v2/images', json={'images': [self.NOPUB_PROT, self.NOPUB_NOPROT]}, status_code=200)
    ret = self.api.image_list(public=True)
    self.assertEqual([self.NOPUB_PROT, self.NOPUB_NOPROT], ret)