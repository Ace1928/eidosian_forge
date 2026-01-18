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
def test_image_change_adding_invalid_location_metadata(self):
    self.assertEqual(2, len(self.store_api.data.keys()))
    context = glance.context.RequestContext(user=USER1)
    image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
    image_stub2 = ImageStub('fake_image_id', status='queued', locations=[])
    image2 = glance.location.ImageProxy(image_stub2, context, self.store_api, self.store_utils)
    location_bad = {'url': UUID2, 'metadata': b'a invalid metadata'}
    self.assertRaises(glance_store.BackendException, image2.locations.__iadd__, [location_bad])
    self.assertEqual([], image_stub2.locations)
    self.assertEqual([], image2.locations)
    image1.delete()
    image2.delete()
    self.assertEqual(2, len(self.store_api.data.keys()))
    self.assertNotIn(UUID2, self.store_api.data.keys())