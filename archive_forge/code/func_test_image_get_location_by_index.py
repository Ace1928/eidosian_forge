from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_image_get_location_by_index(self):
    UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
    self.assertEqual(2, len(self.store_api.data.keys()))
    context = glance.context.RequestContext(user=USER1)
    image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
    image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
    image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
    image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
    location2 = {'url': UUID2, 'metadata': {}}
    location3 = {'url': UUID3, 'metadata': {}}
    with mock.patch('glance.location.store') as mock_store:
        mock_store.get_size_from_uri_and_backend.return_value = 4
        image3.locations += [location2, location3]
    self.assertEqual(1, image_stub3.locations.index(location3))
    self.assertEqual(location2, image_stub3.locations[0])
    image3.delete()
    self.assertEqual(2, len(self.store_api.data.keys()))
    self.assertNotIn(UUID2, self.store_api.data.keys())
    self.assertNotIn(UUID3, self.store_api.data.keys())
    image1.delete()
    image2.delete()